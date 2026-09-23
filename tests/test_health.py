from __future__ import annotations

import asyncio

from tiered_llm import (
    BreakerConfig,
    BreakerRegistry,
    BreakerState,
    DecisionLog,
    HealthMonitor,
    iter_decisions,
)
from tiered_llm.testing import ScriptedProvider


def setup(clock, **providers_health):
    registry = BreakerRegistry(BreakerConfig(failure_threshold=1, recovery_timeout=30), clock=clock)
    targets = []
    for label, healthy in providers_health.items():
        provider = ScriptedProvider(label, healthy=healthy)
        targets.append((provider, registry.get(label)))
    return registry, targets


async def test_probes_only_providers_due_for_a_trial(clock):
    registry, targets = setup(clock, up=True, down=False, fine=True)
    registry.get("up").trip()
    registry.get("down").trip()
    monitor = HealthMonitor(targets)
    assert await monitor.run_once() == {}  # still cooling down
    clock.advance(30)
    assert await monitor.run_once() == {"up": True, "down": False}
    assert registry.get("up").state is BreakerState.CLOSED
    snap = registry.get("down").snapshot()
    assert snap.state is BreakerState.OPEN and snap.retry_after == 60  # backed off
    assert targets[2][0].health_checks == 0  # closed circuits are never probed


async def test_probe_timeout_counts_as_unhealthy(clock):
    registry, targets = setup(clock, hung=True)

    async def never():
        await asyncio.sleep(10)
        return True

    targets[0][0].health_check = never
    registry.get("hung").trip(0)
    monitor = HealthMonitor(targets, probe_timeout=0.01)
    assert await monitor.run_once() == {"hung": False}


async def test_background_loop_closes_recovered_circuits(tmp_path):
    registry = BreakerRegistry(BreakerConfig(failure_threshold=1, recovery_timeout=0))
    provider = ScriptedProvider("p", healthy=True)
    breaker = registry.get("p")
    breaker.trip(0.01)
    log = DecisionLog(tmp_path / "d.jsonl", fsync=False)
    closed = asyncio.Event()
    breaker.add_listener(lambda name, old, new: new is BreakerState.CLOSED and closed.set())
    async with HealthMonitor([(provider, breaker)], interval=0.01, decision_log=log):
        # Wait for the transition itself, not a number of short sleeps: on Windows,
        # sleeps below the ~15 ms timer resolution can return without time passing.
        await asyncio.wait_for(closed.wait(), timeout=5)
    log.close()
    assert breaker.state is BreakerState.CLOSED
    assert any(e["event"] == "health_probe" and e["healthy"] for e in iter_decisions(tmp_path / "d.jsonl"))


async def test_cancelling_a_probe_hands_the_trial_slot_back(clock):
    registry, targets = setup(clock, slow=True)

    async def slow_probe():
        await asyncio.sleep(10)
        return True

    targets[0][0].health_check = slow_probe
    breaker = registry.get("slow")
    breaker.trip(0)
    task = asyncio.create_task(HealthMonitor(targets).run_once())
    await asyncio.sleep(0.01)
    assert breaker.acquire() is None  # the probe holds the only trial slot
    task.cancel()
    await asyncio.gather(task, return_exceptions=True)
    assert breaker.state is BreakerState.HALF_OPEN
    assert breaker.acquire() is not None
