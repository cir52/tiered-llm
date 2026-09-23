from __future__ import annotations

import threading

import pytest

from tiered_llm import BreakerConfig, BreakerRegistry, BreakerState, CircuitBreaker


def make(clock, **overrides) -> CircuitBreaker:
    cfg = BreakerConfig(
        **{"failure_threshold": 3, "recovery_timeout": 30, "max_recovery_timeout": 180, **overrides}
    )
    return CircuitBreaker("p", cfg, clock=clock)


def fail(breaker: CircuitBreaker, times: int = 1) -> None:
    for _ in range(times):
        permit = breaker.acquire()
        assert permit is not None
        permit.failure()


def test_starts_closed_and_admits(clock):
    breaker = make(clock)
    assert breaker.state is BreakerState.CLOSED
    permit = breaker.acquire()
    assert permit is not None and not permit.trial


def test_opens_after_consecutive_failures(clock):
    breaker = make(clock)
    fail(breaker, 2)
    assert breaker.state is BreakerState.CLOSED
    fail(breaker)
    assert breaker.state is BreakerState.OPEN
    assert breaker.acquire() is None
    assert breaker.snapshot().retry_after == pytest.approx(30)


def test_success_resets_the_failure_streak(clock):
    breaker = make(clock)
    fail(breaker, 2)
    breaker.acquire().success()
    fail(breaker, 2)
    assert breaker.state is BreakerState.CLOSED


def test_half_open_admits_a_single_trial(clock):
    breaker = make(clock)
    fail(breaker, 3)
    clock.advance(29.9)
    assert breaker.acquire() is None
    clock.advance(0.1)
    trial = breaker.acquire()
    assert trial is not None and trial.trial
    assert breaker.state is BreakerState.HALF_OPEN
    assert breaker.acquire() is None  # only one probe at a time


def test_successful_trial_closes(clock):
    breaker = make(clock)
    fail(breaker, 3)
    clock.advance(30)
    breaker.acquire().success()
    snap = breaker.snapshot()
    assert snap.state is BreakerState.CLOSED
    assert snap.recovery_attempts == 0 and snap.consecutive_failures == 0


def test_failed_trials_back_off_exponentially_up_to_the_cap(clock):
    breaker = make(clock)
    fail(breaker, 3)
    expected = [60, 120, 180, 180]
    for timeout in expected:
        clock.advance(breaker.snapshot().retry_after)
        breaker.acquire().failure()
        assert breaker.state is BreakerState.OPEN
        assert breaker.snapshot().retry_after == pytest.approx(timeout)


def test_retry_after_opens_immediately_and_is_capped(clock):
    breaker = make(clock)
    breaker.acquire().failure(retry_after=12)
    assert breaker.state is BreakerState.OPEN
    assert breaker.snapshot().retry_after == pytest.approx(12)
    breaker.reset()
    breaker.acquire().failure(retry_after=3600)
    assert breaker.snapshot().retry_after == pytest.approx(180)


def test_stale_results_do_not_move_the_state(clock):
    breaker = make(clock)
    slow = breaker.acquire()  # taken while closed
    fail(breaker, 3)  # meanwhile the provider trips
    slow.success()  # old evidence arrives late
    assert breaker.state is BreakerState.OPEN
    assert breaker.snapshot().total_successes == 1


def test_release_frees_the_trial_slot_without_an_outcome(clock):
    breaker = make(clock)
    fail(breaker, 3)
    clock.advance(30)
    trial = breaker.acquire()
    trial.release()
    assert breaker.state is BreakerState.HALF_OPEN
    assert breaker.acquire() is not None


def test_permit_resolves_only_once(clock):
    breaker = make(clock, failure_threshold=1)
    permit = breaker.acquire()
    permit.success()
    permit.failure()
    assert breaker.state is BreakerState.CLOSED


def test_listeners_see_every_transition_and_cannot_break_calls(clock):
    breaker = make(clock)
    seen = []
    breaker.add_listener(lambda name, old, new: seen.append((old.value, new.value)))
    breaker.add_listener(lambda *_: 1 / 0)
    fail(breaker, 3)
    clock.advance(30)
    breaker.acquire().success()
    assert seen == [("closed", "open"), ("open", "half_open"), ("half_open", "closed")]


def test_manual_trip_and_reset(clock):
    breaker = make(clock)
    breaker.trip(10)
    assert breaker.acquire() is None
    breaker.reset()
    assert breaker.acquire() is not None


def test_explicit_trip_is_not_capped(clock):
    breaker = make(clock)
    breaker.trip(3600)  # planned maintenance
    clock.advance(1000)
    assert breaker.acquire() is None
    assert breaker.snapshot().retry_after == pytest.approx(2600)


def test_config_validation():
    with pytest.raises(ValueError):
        BreakerConfig(failure_threshold=0)
    with pytest.raises(ValueError):
        BreakerConfig(recovery_timeout=100, max_recovery_timeout=10)


def test_registry_shares_breakers_and_applies_listeners_to_new_ones(clock):
    registry = BreakerRegistry(BreakerConfig(failure_threshold=1), clock=clock)
    events = []
    registry.add_listener(lambda name, old, new: events.append(name))
    assert registry.get("a") is registry.get("a")
    registry.get("b").acquire().failure()
    assert events == ["b"]
    custom = registry.get("c", BreakerConfig(failure_threshold=5))
    assert custom.config.failure_threshold == 5
    assert registry.get("c") is custom
    with pytest.raises(ValueError, match="already exists"):
        registry.get("c", BreakerConfig(failure_threshold=2))
    assert {s.name for s in registry.snapshots()} == {"a", "b", "c"}


def test_totals_are_consistent_under_thread_contention():
    breaker = CircuitBreaker("p", BreakerConfig(failure_threshold=10**9))

    def worker() -> None:
        for i in range(2000):
            permit = breaker.acquire()
            assert permit is not None
            permit.success() if i % 2 else permit.failure()

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    snap = breaker.snapshot()
    assert snap.total_successes == snap.total_failures == 8000
