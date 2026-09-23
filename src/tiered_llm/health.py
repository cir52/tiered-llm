"""Background recovery probes.

Without probes, an open circuit only closes when real traffic happens to hit
its half-open window - so the first request after an outage pays for the
experiment. The monitor probes providers whose circuit is not closed, using
their cheapest health check, and closes the circuit before real work arrives.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from collections.abc import Iterable
from types import TracebackType

from .breaker import BreakerState, CircuitBreaker
from .decision_log import DecisionLog
from .providers.base import Provider

__all__ = ["HealthMonitor"]

logger = logging.getLogger("tiered_llm.health")


class HealthMonitor:
    def __init__(
        self,
        targets: Iterable[tuple[Provider, CircuitBreaker]],
        *,
        interval: float = 15.0,
        probe_timeout: float = 12.0,
        decision_log: DecisionLog | None = None,
    ) -> None:
        self.targets = list(targets)
        self.interval = interval
        self.probe_timeout = probe_timeout
        self.decision_log = decision_log
        self._task: asyncio.Task[None] | None = None

    async def run_once(self) -> dict[str, bool]:
        """Probe every provider that is due for a recovery trial. Returns label -> healthy."""
        due = []
        for provider, breaker in self.targets:
            if breaker.state is BreakerState.CLOSED:
                continue
            permit = breaker.acquire()
            if permit is not None:
                due.append((provider, permit))
        outcome: dict[str, bool] = {}
        try:
            results = await asyncio.gather(*(self._probe(p) for p, _ in due))
            for (provider, permit), healthy in zip(due, results, strict=True):
                if healthy:
                    permit.success()
                else:
                    permit.failure()
                outcome[provider.label] = healthy
                if self.decision_log is not None:
                    self.decision_log.log("health_probe", provider=provider.label, healthy=healthy)
        finally:
            # Cancelled mid-probe (stop(), shutdown): hand the trial slots back,
            # otherwise the circuits would stay half-open with no trial allowed.
            for _, permit in due:
                permit.release()
        return outcome

    async def _probe(self, provider: Provider) -> bool:
        try:
            return await asyncio.wait_for(provider.health_check(), self.probe_timeout)
        except asyncio.TimeoutError:
            return False
        except Exception:
            logger.exception("health check of %s raised", provider.label)
            return False

    async def _loop(self) -> None:
        while True:
            await asyncio.sleep(self.interval)
            try:
                await self.run_once()
            except Exception:  # keep the monitor alive whatever happens
                logger.exception("health monitor iteration failed")

    def start(self) -> asyncio.Task[None]:
        if self._task is None or self._task.done():
            self._task = asyncio.create_task(self._loop(), name="tiered-llm-health-monitor")
        return self._task

    async def stop(self) -> None:
        if self._task is not None:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task
            self._task = None

    async def __aenter__(self) -> HealthMonitor:
        self.start()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        await self.stop()
