from __future__ import annotations

import asyncio

import pytest

from tiered_llm import (
    AllProvidersFailedError,
    BreakerConfig,
    BreakerRegistry,
    BreakerState,
    CompletionRequest,
    DecisionLog,
    FallbackChain,
    InvalidRequestError,
    Pricing,
    ProviderUnavailableError,
    RateLimitError,
    RetryPolicy,
    Router,
    Usage,
    iter_decisions,
)
from tiered_llm.testing import ScriptedProvider

REQ = CompletionRequest.from_prompt("hello", metadata={"request_id": "r-1"})


async def test_primary_answers():
    primary, backup = ScriptedProvider("primary", ["from primary"]), ScriptedProvider("backup")
    response = await FallbackChain([primary, backup]).complete(REQ)
    assert response.text == "from primary"
    assert not response.fell_back
    assert [a.outcome for a in response.attempts] == ["ok"]
    assert backup.calls == []


async def test_falls_back_on_retryable_error():
    primary = ScriptedProvider("primary", [ProviderUnavailableError("503")])
    backup = ScriptedProvider("backup", ["from backup"])
    response = await FallbackChain([primary, backup]).complete(REQ)
    assert response.text == "from backup"
    assert response.fell_back
    assert [(a.provider, a.outcome) for a in response.attempts] == [("primary", "error"), ("backup", "ok")]


async def test_open_circuit_is_skipped_without_a_call(clock):
    primary = ScriptedProvider("primary", [ProviderUnavailableError("down")])
    backup = ScriptedProvider("backup")
    chain = FallbackChain(
        [primary, backup], registry=BreakerRegistry(BreakerConfig(failure_threshold=2), clock=clock)
    )
    for _ in range(2):
        await chain.complete(REQ)
    assert chain.breaker(primary).state is BreakerState.OPEN
    response = await chain.complete(REQ)
    assert len(primary.calls) == 2
    assert response.attempts[0].outcome == "skipped"
    assert "circuit open" in response.attempts[0].error


async def test_invalid_request_is_not_retried_and_does_not_trip():
    primary = ScriptedProvider("primary", [InvalidRequestError("prompt too long")])
    backup = ScriptedProvider("backup")
    chain = FallbackChain([primary, backup], breaker_config=BreakerConfig(failure_threshold=1))
    with pytest.raises(InvalidRequestError):
        await chain.complete(REQ)
    assert backup.calls == []
    assert chain.breaker(primary).state is BreakerState.CLOSED


async def test_all_failed_reports_every_attempt():
    chain = FallbackChain(
        [
            ScriptedProvider("a", [ProviderUnavailableError("503")]),
            ScriptedProvider("b", [RateLimitError("slow down")]),
        ],
        name="decide",
    )
    with pytest.raises(AllProvidersFailedError) as info:
        await chain.complete(REQ)
    assert info.value.chain == "decide"
    assert [a.provider for a in info.value.attempts] == ["a", "b"]
    assert "RateLimitError" in info.value.attempts[1].error


async def test_retry_policy_retries_the_same_provider_first():
    flaky = ScriptedProvider("flaky", [ProviderUnavailableError("blip"), "second time lucky"])
    chain = FallbackChain([flaky], retry=RetryPolicy(max_retries=2, base_delay=0, jitter=False))
    response = await chain.complete(REQ)
    assert response.text == "second time lucky"
    assert len(flaky.calls) == 2


async def test_rate_limit_with_retry_after_opens_at_once_and_is_not_retried():
    limited = ScriptedProvider("limited", [RateLimitError("429", retry_after=20)])
    backup = ScriptedProvider("backup")
    chain = FallbackChain([limited, backup], retry=RetryPolicy(max_retries=3, base_delay=0))
    await chain.complete(REQ)
    assert len(limited.calls) == 1
    snap = chain.breaker(limited).snapshot()
    assert snap.state is BreakerState.OPEN and snap.retry_after == pytest.approx(20, abs=1)


async def test_attempt_timeout_counts_as_unavailable():
    slow = ScriptedProvider("slow", ["too late"], latency=1.0)
    fast = ScriptedProvider("fast", ["in time"])
    chain = FallbackChain([slow, fast], attempt_timeout=0.05)
    response = await chain.complete(REQ)
    assert response.text == "in time"
    assert "timed out" in response.attempts[0].error


async def test_cancellation_releases_the_trial_slot(clock):
    slow = ScriptedProvider("slow", ["x"], latency=10)
    chain = FallbackChain([slow], registry=BreakerRegistry(BreakerConfig(failure_threshold=1), clock=clock))
    breaker = chain.breaker(slow)
    breaker.trip(5)
    clock.advance(5)
    task = asyncio.create_task(chain.complete(REQ))
    await asyncio.sleep(0.01)
    assert breaker.acquire() is None  # the in-flight trial holds the only slot
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert breaker.state is BreakerState.HALF_OPEN
    assert breaker.acquire() is not None


async def test_unexpected_exceptions_propagate_and_do_not_count():
    buggy = ScriptedProvider("buggy", [KeyError("bug in my code")])
    chain = FallbackChain(
        [buggy, ScriptedProvider("backup")], breaker_config=BreakerConfig(failure_threshold=1)
    )
    with pytest.raises(KeyError):
        await chain.complete(REQ)
    assert chain.breaker(buggy).state is BreakerState.CLOSED


async def test_duplicate_labels_are_rejected():
    with pytest.raises(ValueError):
        FallbackChain([ScriptedProvider("x"), ScriptedProvider("x")])


async def test_decision_log_records_calls_and_transitions(tmp_path):
    log_path = tmp_path / "decisions.jsonl"
    log = DecisionLog(log_path, fsync=False)
    primary = ScriptedProvider("primary", [ProviderUnavailableError("503")])
    backup = ScriptedProvider(
        "backup", ["ok"], usage=Usage(1000, 500), pricing=Pricing(input_per_mtok=3, output_per_mtok=15)
    )
    chain = FallbackChain(
        [primary, backup], name="decide", breaker_config=BreakerConfig(failure_threshold=1), decision_log=log
    )
    response = await chain.complete(REQ)
    log.close()
    assert response.cost_usd == pytest.approx(0.0105)
    events = list(iter_decisions(log_path))
    calls = [e for e in events if e["event"] == "llm_call"]
    assert [(c["provider"], c["outcome"], c["position"]) for c in calls] == [
        ("primary", "error", 0),
        ("backup", "ok", 1),
    ]
    assert calls[1]["cost_usd"] == pytest.approx(0.0105)
    assert calls[1]["meta"] == {"request_id": "r-1"}
    assert "messages" not in calls[1]  # content logging is opt-in
    transitions = [e for e in events if e["event"] == "breaker_transition"]
    assert transitions and transitions[0]["provider"] == "primary" and transitions[0]["new"] == "open"


async def test_content_logging_is_opt_in(tmp_path):
    log = DecisionLog(tmp_path / "d.jsonl", fsync=False)
    chain = FallbackChain([ScriptedProvider("p", ["answer"])], decision_log=log, log_content=True)
    await chain.complete(REQ)
    log.close()
    (call,) = iter_decisions(tmp_path / "d.jsonl")
    assert call["response"] == "answer"
    assert call["messages"] == [{"role": "user", "content": "hello"}]


async def test_router_shares_one_breaker_per_provider_across_tasks():
    claude = ScriptedProvider("claude", [ProviderUnavailableError("outage")])
    local = ScriptedProvider("local", ["local answer"])
    router = Router(
        {"decide": [claude, local], "summarize": [claude, local]},
        breaker_config=BreakerConfig(failure_threshold=1, recovery_timeout=60),
    )
    await router.complete("decide", REQ)
    response = await router.complete("summarize", REQ)
    assert len(claude.calls) == 1  # the outage found by "decide" is respected by "summarize"
    assert response.attempts[0].outcome == "skipped"
    assert router.status()["claude"]["state"] == "open"
    assert local.calls[-1].metadata["task"] == "summarize"


async def test_router_unknown_task():
    router = Router({"a": [ScriptedProvider("p")]})
    with pytest.raises(KeyError):
        await router.complete("b", REQ)


async def test_router_rejects_two_objects_with_one_label():
    with pytest.raises(ValueError):
        Router({"a": [ScriptedProvider("p")], "b": [ScriptedProvider("p")]})


async def test_transitions_are_logged_once_with_a_shared_registry(tmp_path):
    log = DecisionLog(tmp_path / "d.jsonl", fsync=False)
    registry = BreakerRegistry(BreakerConfig(failure_threshold=1))
    down = ScriptedProvider("down", [ProviderUnavailableError("503")])
    backup = ScriptedProvider("backup")
    FallbackChain([down, backup], name="a", registry=registry, decision_log=log)
    chain_b = FallbackChain([down, backup], name="b", registry=registry, decision_log=log)
    await chain_b.complete(REQ)
    log.close()
    transitions = [e for e in iter_decisions(tmp_path / "d.jsonl") if e["event"] == "breaker_transition"]
    assert len(transitions) == 1


async def test_skip_reason_during_a_recovery_trial(clock):
    slow = ScriptedProvider("slow", ["x"], latency=0.05)
    backup = ScriptedProvider("backup")
    chain = FallbackChain(
        [slow, backup], registry=BreakerRegistry(BreakerConfig(failure_threshold=1), clock=clock)
    )
    chain.breaker(slow).trip(1)
    clock.advance(1)
    trial, other = await asyncio.gather(chain.complete(REQ), chain.complete(REQ))
    assert trial.provider == "slow"
    assert other.attempts[0].error == "recovery trial in progress"
