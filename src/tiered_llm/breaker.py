"""Per-provider circuit breaker.

State machine::

    CLOSED --(failure_threshold consecutive failures)--> OPEN
    OPEN   --(recovery timeout elapsed)----------------> HALF_OPEN
    HALF_OPEN --(trial call succeeds)------------------> CLOSED
    HALF_OPEN --(trial call fails)---------------------> OPEN, with a longer timeout

The recovery timeout grows by ``backoff_multiplier`` after every failed trial
and is capped at ``max_recovery_timeout``, so a provider that is down for an
hour is probed a handful of times instead of on every request.

A 429 carrying ``Retry-After`` opens the circuit immediately for that long
(capped): the provider has told us when to come back, so there is no point in
spending more requests to find out.

Every call takes a :class:`Permit`. The permit remembers the breaker's
*generation* at acquisition time, so a slow call that returns after the
breaker has already moved on (tripped, recovered, been reset) cannot drag it
back into a stale state.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from enum import Enum

__all__ = [
    "BreakerConfig",
    "BreakerRegistry",
    "BreakerSnapshot",
    "BreakerState",
    "CircuitBreaker",
    "Permit",
    "StateListener",
]

logger = logging.getLogger("tiered_llm.breaker")

Clock = Callable[[], float]


class BreakerState(str, Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


StateListener = Callable[[str, BreakerState, BreakerState], None]


@dataclass(frozen=True, slots=True)
class BreakerConfig:
    failure_threshold: int = 3
    """Consecutive failures that open the circuit."""
    recovery_timeout: float = 30.0
    """Seconds the circuit stays open before the first trial call."""
    backoff_multiplier: float = 2.0
    """Factor applied to the recovery timeout after each failed trial."""
    max_recovery_timeout: float = 180.0
    """Upper bound for the recovery timeout and for honoured ``Retry-After`` values."""
    half_open_max_calls: int = 1
    """Concurrent trial calls allowed while half-open."""

    def __post_init__(self) -> None:
        if self.failure_threshold < 1:
            raise ValueError("failure_threshold must be >= 1")
        if self.recovery_timeout < 0 or self.max_recovery_timeout < self.recovery_timeout:
            raise ValueError("need 0 <= recovery_timeout <= max_recovery_timeout")
        if self.backoff_multiplier < 1:
            raise ValueError("backoff_multiplier must be >= 1")
        if self.half_open_max_calls < 1:
            raise ValueError("half_open_max_calls must be >= 1")


@dataclass(frozen=True, slots=True)
class BreakerSnapshot:
    name: str
    state: BreakerState
    consecutive_failures: int
    recovery_attempts: int
    retry_after: float
    """Seconds until the next trial is allowed; 0 when calls are admitted now."""
    total_successes: int
    total_failures: int

    def as_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "state": self.state.value,
            "consecutive_failures": self.consecutive_failures,
            "recovery_attempts": self.recovery_attempts,
            "retry_after": round(self.retry_after, 3),
            "total_successes": self.total_successes,
            "total_failures": self.total_failures,
        }


class Permit:
    """Admission ticket for one call. Resolve it exactly once."""

    __slots__ = ("_breaker", "_done", "_generation", "trial")

    def __init__(self, breaker: CircuitBreaker, generation: int, trial: bool) -> None:
        self._breaker = breaker
        self._generation = generation
        self._done = False
        self.trial = trial
        """True if this call is a half-open recovery trial."""

    def success(self) -> None:
        if not self._done:
            self._done = True
            self._breaker._resolve(self, "success", None)

    def failure(self, *, retry_after: float | None = None) -> None:
        if not self._done:
            self._done = True
            self._breaker._resolve(self, "failure", retry_after)

    def release(self) -> None:
        """Give the permit back without an outcome (cancelled, or the request itself was bad)."""
        if not self._done:
            self._done = True
            self._breaker._resolve(self, "release", None)


class CircuitBreaker:
    def __init__(
        self,
        name: str,
        config: BreakerConfig | None = None,
        *,
        clock: Clock = time.monotonic,
    ) -> None:
        self.name = name
        self.config = config or BreakerConfig()
        self._clock = clock
        self._lock = threading.Lock()
        self._listeners: list[StateListener] = []
        self._state = BreakerState.CLOSED
        self._generation = 0
        self._opened_at = 0.0
        self._open_for = 0.0
        self._consecutive_failures = 0
        self._recovery_attempts = 0
        self._trials_in_flight = 0
        self._total_successes = 0
        self._total_failures = 0

    # -- public API ---------------------------------------------------------

    def add_listener(self, listener: StateListener) -> None:
        """Call ``listener(name, old_state, new_state)`` on every transition."""
        self._listeners.append(listener)

    @property
    def state(self) -> BreakerState:
        with self._lock:
            transition = self._refresh()
            state = self._state
        self._notify(transition)
        return state

    def acquire(self) -> Permit | None:
        """Return a permit if a call may go through now, else ``None``."""
        with self._lock:
            transition = self._refresh()
            permit: Permit | None = None
            if self._state is BreakerState.CLOSED:
                permit = Permit(self, self._generation, trial=False)
            elif (
                self._state is BreakerState.HALF_OPEN
                and self._trials_in_flight < self.config.half_open_max_calls
            ):
                self._trials_in_flight += 1
                permit = Permit(self, self._generation, trial=True)
        self._notify(transition)
        return permit

    def snapshot(self) -> BreakerSnapshot:
        with self._lock:
            transition = self._refresh()
            retry_after = 0.0
            if self._state is BreakerState.OPEN:
                retry_after = max(0.0, self._opened_at + self._open_for - self._clock())
            snap = BreakerSnapshot(
                name=self.name,
                state=self._state,
                consecutive_failures=self._consecutive_failures,
                recovery_attempts=self._recovery_attempts,
                retry_after=retry_after,
                total_successes=self._total_successes,
                total_failures=self._total_failures,
            )
        self._notify(transition)
        return snap

    def reset(self) -> None:
        """Force the circuit closed (admin action / tests)."""
        with self._lock:
            transition = self._move(BreakerState.CLOSED)
            self._consecutive_failures = 0
            self._recovery_attempts = 0
        self._notify(transition)

    def trip(self, seconds: float | None = None) -> None:
        """Force the circuit open, e.g. for planned maintenance of a provider.

        An explicit duration is honoured as given (not capped by ``max_recovery_timeout``).
        """
        with self._lock:
            if seconds is None:
                transition = self._open(self._current_timeout())
            else:
                transition = self._open(seconds, cap=False)
        self._notify(transition)

    # -- internals ----------------------------------------------------------

    def _current_timeout(self) -> float:
        cfg = self.config
        timeout = cfg.recovery_timeout * (cfg.backoff_multiplier**self._recovery_attempts)
        return float(min(timeout, cfg.max_recovery_timeout))

    def _move(self, new: BreakerState) -> tuple[BreakerState, BreakerState] | None:
        old = self._state
        self._state = new
        self._generation += 1
        self._trials_in_flight = 0
        return (old, new) if old is not new else None

    def _open(self, seconds: float, *, cap: bool = True) -> tuple[BreakerState, BreakerState] | None:
        self._opened_at = self._clock()
        if cap:
            seconds = min(seconds, self.config.max_recovery_timeout)
        self._open_for = max(0.0, seconds)
        return self._move(BreakerState.OPEN)

    def _refresh(self) -> tuple[BreakerState, BreakerState] | None:
        if self._state is BreakerState.OPEN and self._clock() - self._opened_at >= self._open_for:
            return self._move(BreakerState.HALF_OPEN)
        return None

    def _resolve(self, permit: Permit, outcome: str, retry_after: float | None) -> None:
        with self._lock:
            transition = None
            if outcome == "success":
                self._total_successes += 1
            elif outcome == "failure":
                self._total_failures += 1

            current = permit._generation == self._generation
            if permit.trial and current:
                self._trials_in_flight = max(0, self._trials_in_flight - 1)

            if current and outcome == "success":
                self._consecutive_failures = 0
                if self._state is not BreakerState.CLOSED:
                    self._recovery_attempts = 0
                    transition = self._move(BreakerState.CLOSED)
            elif current and outcome == "failure":
                self._consecutive_failures += 1
                if self._state is BreakerState.HALF_OPEN:
                    self._recovery_attempts += 1
                    timeout = self._current_timeout()
                    if retry_after is not None:
                        timeout = max(timeout, retry_after)
                    transition = self._open(timeout)
                elif retry_after is not None and retry_after > 0:
                    transition = self._open(retry_after)
                elif self._consecutive_failures >= self.config.failure_threshold:
                    transition = self._open(self._current_timeout())
            # Results from a stale generation only update the totals above.
        self._notify(transition)

    def _notify(self, transition: tuple[BreakerState, BreakerState] | None) -> None:
        if transition is None:
            return
        old, new = transition
        log = logger.warning if new is BreakerState.OPEN else logger.info
        log("circuit %s: %s -> %s", self.name, old.value, new.value)
        for listener in list(self._listeners):
            try:
                listener(self.name, old, new)
            except Exception:  # a broken listener must never break a call
                logger.exception("breaker listener failed for %s", self.name)


class BreakerRegistry:
    """One breaker per provider label, shared by every chain that uses the provider.

    If Claude is the primary for one task and the fallback for another, both
    chains must see the same circuit: an outage discovered by one is known to all.
    """

    def __init__(self, config: BreakerConfig | None = None, *, clock: Clock = time.monotonic) -> None:
        self.config = config or BreakerConfig()
        self._clock = clock
        self._lock = threading.Lock()
        self._breakers: dict[str, CircuitBreaker] = {}
        self._listeners: list[StateListener] = []
        self._attached_logs: list[object] = []

    def get(self, name: str, config: BreakerConfig | None = None) -> CircuitBreaker:
        """Return the breaker for ``name``, creating it on first use.

        Asking for an existing breaker with a *different* explicit config is an
        error: two chains would otherwise silently disagree about thresholds.
        """
        with self._lock:
            breaker = self._breakers.get(name)
            if breaker is None:
                breaker = CircuitBreaker(name, config or self.config, clock=self._clock)
                for listener in self._listeners:
                    breaker.add_listener(listener)
                self._breakers[name] = breaker
            elif config is not None and config != breaker.config:
                raise ValueError(f"breaker '{name}' already exists with {breaker.config}, requested {config}")
            return breaker

    def _attach_log(self, log: object) -> bool:
        """Remember that ``log`` receives this registry's transitions; False if it already does."""
        with self._lock:
            if any(existing is log for existing in self._attached_logs):
                return False
            self._attached_logs.append(log)
            return True

    def add_listener(self, listener: StateListener) -> None:
        with self._lock:
            self._listeners.append(listener)
            breakers = list(self._breakers.values())
        for breaker in breakers:
            breaker.add_listener(listener)

    def snapshots(self) -> list[BreakerSnapshot]:
        with self._lock:
            breakers = list(self._breakers.values())
        return [b.snapshot() for b in breakers]

    def __iter__(self) -> Iterator[CircuitBreaker]:
        with self._lock:
            return iter(list(self._breakers.values()))

    def __len__(self) -> int:
        return len(self._breakers)
