from __future__ import annotations

import pytest

from tiered_llm import (
    AllProvidersFailedError,
    Cascade,
    CascadeItem,
    CascadeStats,
    CompletionRequest,
    DecisionLog,
    FallbackChain,
    Pricing,
    ProviderUnavailableError,
    Usage,
    iter_decisions,
)
from tiered_llm.testing import ScriptedProvider


def score_gate(response):
    return response.parse_json()["score"] >= 0.7


def screener(scores):
    return ScriptedProvider("local", [lambda req: f'{{"score": {scores[req.metadata["item_id"]]}}}'])


def judge():
    return ScriptedProvider(
        "frontier",
        [lambda req: f"decision for {req.metadata['item_id']}"],
        usage=Usage(1000, 200),
        pricing=Pricing(3, 15),
    )


async def test_low_scores_never_reach_the_expensive_model():
    big = judge()
    cascade = Cascade(screen=screener({"a": 0.2}), decide=big, gate=score_gate)
    result = await cascade.run(
        CompletionRequest.from_prompt("a"), CompletionRequest.from_prompt("x"), item_id="a"
    )
    assert not result.escalated and result.decision is None
    assert result.final is result.screen
    assert big.calls == []


async def test_high_scores_escalate_with_a_request_built_from_the_screen():
    big = judge()
    cascade = Cascade(screen=screener({"b": 0.9}), decide=big, gate=score_gate)
    result = await cascade.run(
        CompletionRequest.from_prompt("b"),
        lambda screened: CompletionRequest.from_prompt(f"screen said {screened.text}"),
        item_id="b",
    )
    assert result.escalated and result.final.text == "decision for b"
    sent = big.calls[0]
    assert "screen said" in sent.messages[0].content
    assert sent.metadata["stage"] == "decide" and sent.metadata["cascade"] == "cascade"
    assert result.cost_usd == pytest.approx((1000 * 3 + 200 * 15) / 1e6)


@pytest.mark.parametrize(("policy", "escalated"), [("reject", False), ("escalate", True)])
async def test_gate_errors_follow_the_policy(policy, escalated):
    garbage = ScriptedProvider("local", ["not json at all"])
    cascade = Cascade(screen=garbage, decide=judge(), gate=score_gate, on_gate_error=policy)
    result = await cascade.run(
        CompletionRequest.from_prompt("q"), CompletionRequest.from_prompt("d"), item_id="q"
    )
    assert result.escalated is escalated
    assert "JSONExtractionError" in result.gate_error


async def test_gate_errors_raise_by_default():
    cascade = Cascade(screen=ScriptedProvider("local", ["nope"]), decide=judge(), gate=score_gate)
    with pytest.raises(ValueError):
        await cascade.run(CompletionRequest.from_prompt("q"), CompletionRequest.from_prompt("d"))


async def test_run_many_keeps_order_and_reports_economics(tmp_path):
    scores = {f"item-{i}": (0.9 if i % 4 == 0 else 0.1) for i in range(20)}
    log = DecisionLog(tmp_path / "d.jsonl", fsync=False)
    cascade = Cascade(
        screen=FallbackChain([screener(scores)], name="screen", decision_log=log),
        decide=FallbackChain([judge()], name="decide", decision_log=log),
        gate=score_gate,
        decision_log=log,
    )
    items = [
        CascadeItem(CompletionRequest.from_prompt(k), CompletionRequest.from_prompt("decide"), item_id=k)
        for k in scores
    ]
    results = await cascade.run_many(items, concurrency=4)
    log.close()
    assert [r.item_id for r in results] == list(scores)
    stats = CascadeStats.from_results(results)
    assert (stats.items, stats.escalated) == (20, 5)
    assert stats.escalation_rate == 0.25
    assert stats.screen_cost_usd == 0 and stats.total_cost_usd == pytest.approx(5 * 0.006)
    events = list(iter_decisions(tmp_path / "d.jsonl"))
    verdicts = [e for e in events if e["event"] == "cascade_verdict"]
    assert len(verdicts) == 20 and sum(v["escalated"] for v in verdicts) == 5
    decide_calls = [e for e in events if e["event"] == "llm_call" and e["chain"] == "decide"]
    assert {e["meta"]["stage"] for e in decide_calls} == {"decide"}


async def test_run_many_can_collect_failures():
    down = ScriptedProvider("frontier", [ProviderUnavailableError("down")])
    cascade = Cascade(screen=screener({"a": 0.9, "b": 0.1}), decide=FallbackChain([down]), gate=score_gate)
    items = [
        CascadeItem(CompletionRequest.from_prompt(k), CompletionRequest.from_prompt("d"), item_id=k)
        for k in ("a", "b")
    ]
    first, second = await cascade.run_many(items, return_exceptions=True)
    assert isinstance(first, AllProvidersFailedError)
    assert second.escalated is False


async def test_run_many_cancels_the_rest_after_a_failure():
    import asyncio

    finished = []

    def slow_answer(request):
        finished.append(request.metadata["item_id"])
        return '{"score": 0.1}'

    def screen(request):
        if request.metadata["item_id"] == "bad":
            raise ProviderUnavailableError("down")
        return slow_answer(request)

    class SlowScreen(ScriptedProvider):
        async def _complete(self, request):
            if request.metadata["item_id"] != "bad":
                await asyncio.sleep(0.2)
            return await super()._complete(request)

    cascade = Cascade(screen=FallbackChain([SlowScreen("local", [screen])]), decide=judge(), gate=score_gate)
    items = [
        CascadeItem(CompletionRequest.from_prompt(k), CompletionRequest.from_prompt("d"), item_id=k)
        for k in ("bad", "a", "b", "c")
    ]
    with pytest.raises(AllProvidersFailedError):
        await cascade.run_many(items, concurrency=4)
    await asyncio.sleep(0.3)
    assert finished == []  # the slow items were cancelled, not left running
