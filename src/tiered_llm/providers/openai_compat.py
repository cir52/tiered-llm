"""OpenAI-compatible chat completions: OpenAI, DeepSeek, OpenRouter, llama.cpp, vLLM ..."""

from __future__ import annotations

from typing import Any, ClassVar

from ..errors import ProviderError, ResponseParseError
from ..types import CompletionRequest, CompletionResponse, Usage
from .base import HTTPProvider, resolve_api_key, with_json_instruction

__all__ = [
    "DeepSeekProvider",
    "LlamaCppProvider",
    "OpenAICompatibleProvider",
    "OpenAIProvider",
    "OpenRouterProvider",
]


class OpenAICompatibleProvider(HTTPProvider):
    """Any server exposing ``POST {base_url}/chat/completions``.

    Subclasses only set defaults. Use this class directly for self-hosted
    servers (vLLM, LM Studio, TGI ...): ``OpenAICompatibleProvider("my-model",
    base_url="http://gpu-box:8000/v1")``.
    """

    kind = "openai_compatible"
    default_base_url: ClassVar[str | None] = None
    default_api_key_env: ClassVar[str | None] = None
    requires_api_key: ClassVar[bool] = False
    max_tokens_field: ClassVar[str] = "max_tokens"

    def __init__(
        self,
        model: str,
        *,
        base_url: str | None = None,
        api_key: str | None = None,
        api_key_env: str | None = None,
        **kwargs: Any,
    ) -> None:
        url = base_url or self.default_base_url
        if not url:
            raise ValueError(f"{self.kind}: base_url is required")
        super().__init__(model=model, base_url=url, **kwargs)
        env = api_key_env or self.default_api_key_env
        if self.requires_api_key:
            self._api_key: str | None = resolve_api_key(self.kind, api_key, env)
        else:
            self._api_key = api_key

    def _auth_headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self._api_key}"} if self._api_key else {}

    def build_payload(self, request: CompletionRequest) -> dict[str, Any]:
        messages: list[dict[str, str]] = []
        system = with_json_instruction(request.system_prompt, request.json_mode)
        if system:
            messages.append({"role": "system", "content": system})
        messages.extend({"role": m.role, "content": m.content} for m in request.conversation)
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            self.max_tokens_field: request.max_tokens,
        }
        if request.temperature is not None:
            payload["temperature"] = request.temperature
        if request.stop:
            payload["stop"] = list(request.stop)
        if request.json_mode:
            payload["response_format"] = {"type": "json_object"}
        return payload

    async def _complete(self, request: CompletionRequest) -> CompletionResponse:
        data = await self._post_json("/chat/completions", self.build_payload(request))
        choices = data.get("choices")
        if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
            # Some gateways (OpenRouter) report upstream failures as 200 + error body.
            error = data.get("error")
            if error:
                message = error.get("message", error) if isinstance(error, dict) else error
                raise ResponseParseError(f"upstream error: {message}", provider=self.label)
            raise ResponseParseError("missing 'choices' in response", provider=self.label)
        choice = choices[0]
        message = choice.get("message") or {}
        usage = data.get("usage") or {}
        return CompletionResponse(
            text=str(message.get("content") or ""),
            provider=self.label,
            model=str(data.get("model") or self.model),
            usage=Usage(int(usage.get("prompt_tokens", 0)), int(usage.get("completion_tokens", 0))),
            finish_reason=choice.get("finish_reason"),
            raw=data,
        )


class OpenAIProvider(OpenAICompatibleProvider):
    kind = "openai"
    default_base_url = "https://api.openai.com/v1"
    default_api_key_env = "OPENAI_API_KEY"
    requires_api_key = True
    max_tokens_field = "max_completion_tokens"


class DeepSeekProvider(OpenAICompatibleProvider):
    kind = "deepseek"
    default_base_url = "https://api.deepseek.com/v1"
    default_api_key_env = "DEEPSEEK_API_KEY"
    requires_api_key = True


class OpenRouterProvider(OpenAICompatibleProvider):
    """OpenRouter - one key, many upstream vendors.

    Useful as the *first* hop in a chain with the vendor's direct API behind it
    ("proxy -> direct"): if the gateway has an incident, the chain goes straight
    to the vendor with a different key and a different network path.
    """

    kind = "openrouter"
    default_base_url = "https://openrouter.ai/api/v1"
    default_api_key_env = "OPENROUTER_API_KEY"
    requires_api_key = True

    def __init__(
        self,
        model: str,
        *,
        app_name: str | None = None,
        app_url: str | None = None,
        **kwargs: Any,
    ) -> None:
        headers = dict(kwargs.pop("headers", None) or {})
        if app_name:
            headers.setdefault("X-Title", app_name)
        if app_url:
            headers.setdefault("HTTP-Referer", app_url)
        super().__init__(model, headers=headers, **kwargs)


class LlamaCppProvider(OpenAICompatibleProvider):
    """llama.cpp ``llama-server`` (OpenAI-compatible endpoint + ``/health``)."""

    kind = "llamacpp"
    default_base_url = "http://localhost:8080/v1"

    def __init__(self, model: str = "local", **kwargs: Any) -> None:
        kwargs.setdefault("timeout", 120.0)
        super().__init__(model, **kwargs)

    async def health_check(self) -> bool:
        root = self.base_url.removesuffix("/v1")
        try:
            await self._request("GET", f"{root}/health", timeout=5.0)
        except ProviderError as exc:
            # Older servers answer 503 "no slot available" when all slots are
            # busy: the server works, it is just saturated. "Loading model"
            # (also 503) is genuinely not ready.
            return "no slot available" in exc.message.lower()
        return True
