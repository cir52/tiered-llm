from __future__ import annotations

import asyncio
import json

import httpx
import pytest

from tiered_llm import (
    AnthropicProvider,
    AuthenticationError,
    BreakerConfig,
    CompletionRequest,
    CompletionResponse,
    DeepSeekProvider,
    GeminiProvider,
    InvalidRequestError,
    LlamaCppProvider,
    Message,
    OllamaProvider,
    OpenAICompatibleProvider,
    OpenAIProvider,
    OpenRouterProvider,
    Pricing,
    Provider,
    ProviderUnavailableError,
    RateLimitError,
    ResponseParseError,
    Router,
    Usage,
    build_provider,
)
from tiered_llm.providers import choose_model


class Recorder:
    """httpx transport that records requests and replies from a handler."""

    def __init__(self, handler):
        self.handler = handler
        self.requests: list[httpx.Request] = []

    def client(self) -> httpx.AsyncClient:
        def handle(request: httpx.Request) -> httpx.Response:
            self.requests.append(request)
            return self.handler(request)

        return httpx.AsyncClient(transport=httpx.MockTransport(handle))

    @property
    def body(self) -> dict:
        return json.loads(self.requests[-1].content)


def reply(payload, status=200, headers=None):
    return lambda request: httpx.Response(status, json=payload, headers=headers)


CHAT = CompletionRequest(
    messages=(
        Message.system("be terse"),
        Message.user("hi"),
        Message.assistant("hello"),
        Message.user("json?"),
    ),
    max_tokens=50,
    temperature=0.2,
    stop=("END",),
)


async def test_anthropic_request_and_response():
    rec = Recorder(
        reply(
            {
                "model": "claude-x",
                "content": [{"type": "text", "text": '{"ok": '}, {"type": "text", "text": "true}"}],
                "usage": {"input_tokens": 12, "output_tokens": 4},
                "stop_reason": "end_turn",
            }
        )
    )
    provider = AnthropicProvider("claude-x", api_key="sk-test", client=rec.client(), pricing=Pricing(3, 15))
    response = await provider.complete(
        CompletionRequest(messages=CHAT.messages, max_tokens=50, json_mode=True, stop=("END",))
    )
    request = rec.requests[0]
    assert request.url == "https://api.anthropic.com/v1/messages"
    assert request.headers["x-api-key"] == "sk-test"
    assert request.headers["anthropic-version"] == "2023-06-01"
    body = rec.body
    assert body["system"].startswith("be terse") and "JSON" in body["system"]
    assert [m["role"] for m in body["messages"]] == ["user", "assistant", "user"]
    assert body["stop_sequences"] == ["END"]
    assert response.parse_json() == {"ok": True}
    assert response.usage == Usage(12, 4)
    assert response.cost_usd == pytest.approx((12 * 3 + 4 * 15) / 1e6)
    assert response.latency_ms > 0
    assert "sk-test" not in repr(provider)


async def test_gemini_request_and_response():
    rec = Recorder(
        reply(
            {
                "candidates": [
                    {
                        "content": {"parts": [{"text": "thinking...", "thought": True}, {"text": "answer"}]},
                        "finishReason": "STOP",
                    }
                ],
                "usageMetadata": {"promptTokenCount": 7, "candidatesTokenCount": 3, "thoughtsTokenCount": 20},
            }
        )
    )
    provider = GeminiProvider("gemini-x", api_key="g-key", client=rec.client())
    response = await provider.complete(
        CompletionRequest(messages=CHAT.messages, max_tokens=50, temperature=0.2, json_mode=True)
    )
    request = rec.requests[0]
    assert request.url.path == "/v1beta/models/gemini-x:generateContent"
    assert request.headers["x-goog-api-key"] == "g-key"
    body = rec.body
    assert [c["role"] for c in body["contents"]] == ["user", "model", "user"]
    assert body["systemInstruction"] == {"parts": [{"text": "be terse"}]}
    assert body["generationConfig"] == {
        "maxOutputTokens": 50,
        "temperature": 0.2,
        "responseMimeType": "application/json",
    }
    assert response.text == "answer"
    assert response.usage == Usage(7, 23)  # thinking tokens are billed as output


async def test_gemini_blocked_prompt_is_a_parse_error():
    rec = Recorder(reply({"promptFeedback": {"blockReason": "SAFETY"}}))
    provider = GeminiProvider("gemini-x", api_key="k", client=rec.client())
    with pytest.raises(ResponseParseError, match="SAFETY"):
        await provider.complete(CHAT)


OPENAI_REPLY = {
    "model": "m",
    "choices": [{"message": {"content": "hey"}, "finish_reason": "stop"}],
    "usage": {"prompt_tokens": 5, "completion_tokens": 2},
}


@pytest.mark.parametrize(
    ("cls", "url", "tokens_field"),
    [
        (OpenAIProvider, "https://api.openai.com/v1/chat/completions", "max_completion_tokens"),
        (DeepSeekProvider, "https://api.deepseek.com/v1/chat/completions", "max_tokens"),
        (OpenRouterProvider, "https://openrouter.ai/api/v1/chat/completions", "max_tokens"),
    ],
)
async def test_openai_compatible_vendors(cls, url, tokens_field):
    rec = Recorder(reply(OPENAI_REPLY))
    provider = cls("m", api_key="key", client=rec.client())
    response = await provider.complete(
        CompletionRequest.from_prompt("x", system="s", json_mode=True, max_tokens=9)
    )
    request = rec.requests[0]
    assert str(request.url) == url
    assert request.headers["authorization"] == "Bearer key"
    body = rec.body
    assert body[tokens_field] == 9
    assert body["response_format"] == {"type": "json_object"}
    assert body["messages"][0]["role"] == "system" and "JSON" in body["messages"][0]["content"]
    assert response.text == "hey" and response.usage == Usage(5, 2)


async def test_openrouter_attribution_headers_and_error_bodies():
    rec = Recorder(reply({"error": {"message": "upstream exploded", "code": 502}}))
    provider = OpenRouterProvider(
        "m", api_key="k", app_name="demo", app_url="https://example.com", client=rec.client()
    )
    with pytest.raises(ResponseParseError, match="upstream exploded"):
        await provider.complete(CHAT)
    assert rec.requests[0].headers["x-title"] == "demo"
    assert rec.requests[0].headers["http-referer"] == "https://example.com"


async def test_self_hosted_server_needs_no_key():
    rec = Recorder(reply(OPENAI_REPLY))
    provider = OpenAICompatibleProvider("m", base_url="http://gpu-box:8000/v1/", client=rec.client())
    await provider.complete(CHAT)
    assert str(rec.requests[0].url) == "http://gpu-box:8000/v1/chat/completions"
    assert "authorization" not in rec.requests[0].headers


async def test_ollama_request_response_and_health():
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/tags":
            return httpx.Response(200, json={"models": [{"name": "qwen3:14b"}, {"name": "gemma3:12b"}]})
        return httpx.Response(
            200,
            json={
                "model": "qwen3:14b",
                "message": {"role": "assistant", "content": '<think>hmm</think>{"score": 0.9}'},
                "prompt_eval_count": 30,
                "eval_count": 8,
                "done_reason": "stop",
            },
        )

    rec = Recorder(handler)
    provider = OllamaProvider("qwen3:14b", client=rec.client(), options={"num_ctx": 8192})
    response = await provider.complete(CompletionRequest.from_prompt("rate", json_mode=True, max_tokens=64))
    body = rec.body
    assert body["stream"] is False and body["format"] == "json" and body["keep_alive"] == "60m"
    assert body["options"] == {"num_ctx": 8192, "num_predict": 64}
    assert response.parse_json() == {"score": 0.9}
    assert response.usage == Usage(30, 8) and response.cost_usd == 0
    assert await provider.health_check() is True
    assert await OllamaProvider("gemma3:12b", client=rec.client()).health_check() is True
    # /api/chat resolves an untagged name to ":latest", so the health check must too:
    assert await OllamaProvider("gemma3", client=rec.client()).health_check() is False
    assert await OllamaProvider("llama3.3", client=rec.client()).health_check() is False


def test_choose_model_prefers_exact_then_prefix():
    installed = ["gemma3:12b", "gemma3", "deepseek-r1:8b"]
    assert choose_model(["qwen3", "gemma3"], installed) == "gemma3"
    assert choose_model(["deepseek-r1"], installed) == "deepseek-r1:8b"
    assert choose_model(["mistral"], installed) is None


async def test_llamacpp_health_uses_root_endpoint():
    rec = Recorder(reply({"status": "ok"}))
    provider = LlamaCppProvider(client=rec.client())
    assert await provider.health_check() is True
    assert str(rec.requests[0].url) == "http://localhost:8080/health"
    busy = Recorder(reply({"status": "no slot available"}, status=503))
    assert await LlamaCppProvider(client=busy.client()).health_check() is True
    loading = Recorder(reply({"error": {"code": 503, "message": "Loading model"}}, status=503))
    assert await LlamaCppProvider(client=loading.client()).health_check() is False


@pytest.mark.parametrize(
    ("status", "headers", "error", "retryable"),
    [
        (429, {"retry-after": "7"}, RateLimitError, True),
        (401, None, AuthenticationError, True),
        (400, None, InvalidRequestError, False),
        (404, None, ProviderUnavailableError, True),
        (529, None, ProviderUnavailableError, True),
        (402, None, ProviderUnavailableError, True),
    ],
)
async def test_http_status_mapping(status, headers, error, retryable):
    rec = Recorder(reply({"error": {"message": "nope"}}, status=status, headers=headers))
    provider = DeepSeekProvider("m", api_key="k", client=rec.client())
    with pytest.raises(error) as info:
        await provider.complete(CHAT)
    assert info.value.retryable is retryable
    assert info.value.status_code == status
    assert "nope" in str(info.value)
    if status == 429:
        assert info.value.retry_after == 7


async def test_transport_errors_and_garbage_bodies():
    def timeout(request):
        raise httpx.ReadTimeout("slow", request=request)

    provider = DeepSeekProvider("m", api_key="k", client=Recorder(timeout).client())
    with pytest.raises(ProviderUnavailableError, match="timed out"):
        await provider.complete(CHAT)

    def refused(request):
        raise httpx.ConnectError("refused", request=request)

    provider = DeepSeekProvider("m", api_key="k", client=Recorder(refused).client())
    with pytest.raises(ProviderUnavailableError, match="ConnectError"):
        await provider.complete(CHAT)

    garbage = Recorder(lambda r: httpx.Response(200, text="<html>gateway</html>"))
    provider = DeepSeekProvider("m", api_key="k", client=garbage.client())
    with pytest.raises(ResponseParseError):
        await provider.complete(CHAT)


def test_api_keys_come_from_argument_or_environment(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(ValueError, match="ANTHROPIC_API_KEY"):
        AnthropicProvider("m")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "from-env")
    assert AnthropicProvider("m")._api_key == "from-env"
    monkeypatch.setenv("MY_KEY", "custom")
    assert GeminiProvider("m", api_key_env="MY_KEY")._api_key == "custom"


async def test_max_concurrency_serialises_calls():
    class Slow(Provider):
        kind = "slow"
        active = peak = 0

        async def _complete(self, request):
            Slow.active += 1
            Slow.peak = max(Slow.peak, Slow.active)
            await asyncio.sleep(0.01)
            Slow.active -= 1
            return CompletionResponse(text="x", provider=self.label, model=self.model)

    provider = Slow(model="m", max_concurrency=2)
    await asyncio.gather(*(provider.complete(CHAT) for _ in range(10)))
    assert Slow.peak == 2


def test_limiter_works_across_event_loops():
    provider = OllamaProvider("m", client=None)
    for _ in range(3):  # a plain asyncio.Semaphore would be bound to the first loop
        asyncio.run(_use_limiter(provider))


async def _use_limiter(provider):
    async def hold():
        async with provider._limiter:
            await asyncio.sleep(0.001)

    await asyncio.gather(hold(), hold(), hold())


def test_build_provider_from_config(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "k")
    provider = build_provider(
        "claude",
        {
            "type": "anthropic",
            "model": "claude-x",
            "pricing": {"input_per_mtok": 3, "output_per_mtok": 15},
            "breaker": {"failure_threshold": 5},
            "timeout": 20,
        },
    )
    assert isinstance(provider, AnthropicProvider)
    assert provider.label == "claude" and provider.timeout == 20
    assert provider.pricing == Pricing(3, 15)
    assert provider.breaker_config == BreakerConfig(failure_threshold=5)
    with pytest.raises(ValueError, match="api_key_env"):
        build_provider("x", {"type": "anthropic", "model": "m", "api_key": "sk-oops"})
    with pytest.raises(ValueError, match="unknown type"):
        build_provider("x", {"type": "nope", "model": "m"})


def test_router_from_config(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "k")
    router = Router.from_config(
        {
            "breaker": {"failure_threshold": 2},
            "providers": {
                "local": {"type": "ollama", "model": "qwen3:14b"},
                "sonnet": {
                    "type": "openrouter",
                    "model": "anthropic/claude-x",
                    "breaker": {"failure_threshold": 4},
                },
            },
            "routes": {"screen": ["local"], "decide": ["sonnet", "local"]},
        }
    )
    assert [p.label for p in router.chain("decide").providers] == ["sonnet", "local"]
    assert router.chain("decide").breaker("local").config.failure_threshold == 2
    assert router.chain("decide").breaker("sonnet").config.failure_threshold == 4
    with pytest.raises(ValueError, match="unknown providers"):
        Router.from_config({"providers": {}, "routes": {"x": ["missing"]}})


async def test_default_health_check_is_a_one_token_completion():
    rec = Recorder(reply(OPENAI_REPLY))
    provider = DeepSeekProvider("m", api_key="k", client=rec.client())
    assert await provider.health_check() is True
    assert rec.body["max_tokens"] == 1
    down = DeepSeekProvider("m", api_key="k", client=Recorder(reply({}, status=503)).client())
    assert await down.health_check() is False


def test_retry_after_accepts_seconds_and_http_dates():
    from datetime import datetime, timedelta, timezone
    from email.utils import format_datetime

    from tiered_llm.providers.base import _parse_retry_after

    assert _parse_retry_after("12") == 12
    assert _parse_retry_after(None) is None
    assert _parse_retry_after("soon") is None
    future = format_datetime(datetime.now(timezone.utc) + timedelta(seconds=60), usegmt=True)
    assert 55 < _parse_retry_after(future) <= 60


async def test_owned_client_is_closed():
    provider = DeepSeekProvider("m", api_key="k")
    client = provider._client_or_create()
    await provider.aclose()
    assert client.is_closed


async def test_vendor_quirks_in_400_responses():
    low_credit = Recorder(
        reply({"type": "error", "error": {"message": "Your credit balance is too low"}}, status=400)
    )
    with pytest.raises(ProviderUnavailableError) as info:
        await AnthropicProvider("m", api_key="k", client=low_credit.client()).complete(CHAT)
    assert info.value.retryable

    bad_key = Recorder(
        reply(
            {
                "error": {
                    "code": 400,
                    "message": "API key not valid. Please pass a valid API key.",
                    "status": "INVALID_ARGUMENT",
                    "details": [{"reason": "API_KEY_INVALID"}],
                }
            },
            status=400,
        )
    )
    with pytest.raises(AuthenticationError):
        await GeminiProvider("m", api_key="k", client=bad_key.client()).complete(CHAT)

    plain_400 = Recorder(reply({"error": {"message": "max_tokens too large"}}, status=400))
    with pytest.raises(InvalidRequestError):
        await AnthropicProvider("m", api_key="k", client=plain_400.client()).complete(CHAT)


async def test_gemini_safety_block_is_a_failure_not_an_empty_answer():
    rec = Recorder(reply({"candidates": [{"finishReason": "SAFETY", "content": {"parts": []}}]}))
    with pytest.raises(ResponseParseError, match="SAFETY"):
        await GeminiProvider("m", api_key="k", client=rec.client()).complete(CHAT)


async def test_queueing_for_a_busy_gpu_does_not_count_against_the_timeout():
    from tiered_llm import BreakerState, FallbackChain
    from tiered_llm.testing import ScriptedProvider

    gpu = ScriptedProvider("gpu", ["ok"], latency=0.05, max_concurrency=1)
    chain = FallbackChain([gpu], attempt_timeout=0.2)
    results = await asyncio.gather(*(chain.complete(CHAT) for _ in range(8)))
    assert len(results) == 8  # 8 x 0.05 s queued > 0.2 s, yet nothing timed out
    assert chain.breaker(gpu).state is BreakerState.CLOSED


def test_owned_http_clients_are_per_event_loop():
    provider = DeepSeekProvider("m", api_key="k")

    async def grab():
        return provider._client_or_create()

    first = asyncio.run(grab())
    second = asyncio.run(grab())
    assert first is not second


async def test_caller_supplied_client_is_left_open():
    client = httpx.AsyncClient()
    provider = DeepSeekProvider("m", api_key="k", client=client)
    await provider.aclose()
    assert not client.is_closed
    await client.aclose()
