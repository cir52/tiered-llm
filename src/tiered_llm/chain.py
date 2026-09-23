"""Ordered fallback across providers, each guarded by its own circuit breaker."""

from __future__ import annotations

import asyncio
import random
import time
from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import Any

from .breaker import BreakerConfig, BreakerRegistry, BreakerSnapshot, BreakerState, CircuitBreaker
from .decision_log import DecisionLog
from .errors import AllProvidersFailedError, ProviderError, RateLimitError
from .providers.base import Provider
from .types import Attempt, CompletionRequest, CompletionResponse

__all__ = ["FallbackChain", "RetryPolicy", "log_breaker_transitions"]


@dataclass(frozen=True, slots=True)
class RetryPolicy:
    """Retries on the *same* provider before moving on. Off by default:
    in a chain, the next provider usually is the better retry."""

    max_retries: int = 0
    base_delay: float = 0.5
    max_delay: float = 5.0
    jitter: bool = True

    def delay(self, retry_number: int) -> float:
        delay: float = min(self.base_delay * (2.0 ** (retry_number - 1)), self.max_delay)
        return delay * (0.5 + random.random()) if self.jitter else delay


def log_breaker_transitions(registry: BreakerRegistry, log: DecisionLog) -> None:
    """Write every breaker transition in ``registry`` to ``log`` (idempotent per log)."""
    if not registry._attach_log(log):
        return

    def listener(name: str, old: BreakerState, new: BreakerState) -> None:
        breaker = registry.get(name)
        snap = breaker.snapshot()
        log.log(
            "breaker_transition",
            provider=name,
            old=old.value,
            new=new.value,
            consecutive_failures=snap.consecutive_failures,
            recovery_attempts=snap.recovery_attempts,
            retry_after=round(snap.retry_after, 3),
        )

    registry.add_listener(listener)


class FallbackChain:
    """Send a request to the first provider that is up; fall through on failure.

    * A provider whose circuit is open is skipped without a network call.
    * Retryable errors are counted by that provider's breaker and the chain
      moves on; non-retryable ones (a malformed request) are re-raised at once.
    * Every attempt, including skips, is attached to the response and written
      to the decision log.
    """

    def __init__(
        self,
        providers: Sequence[Provider],
        *,
        name: str = "default",
        registry: BreakerRegistry | None = None,
        breaker_config: BreakerConfig | None = None,
        retry: RetryPolicy | None = None,
        attempt_timeout: float | None = None,
        decision_log: DecisionLog | None = None,
        log_content: bool = False,
    ) -> None:
        if not providers:
            raise ValueError("a chain needs at least one provider")
        labels = [p.label for p in providers]
        if len(set(labels)) != len(labels):
            raise ValueError(f"duplicate provider labels in chain '{name}': {labels}")
        self.name = name
        self.providers = tuple(providers)
        self.retry = retry or RetryPolicy()
        self.attempt_timeout = attempt_timeout
        self.decision_log = decision_log
        self.log_content = log_content
        self.registry = registry if registry is not None else BreakerRegistry(breaker_config)
        self._breakers: dict[str, CircuitBreaker] = {
            p.label: self.registry.get(p.label, p.breaker_config) for p in self.providers
        }
        if decision_log is not None:
            log_breaker_transitions(self.registry, decision_log)

    def breaker(self, provider: Provider | str) -> CircuitBreaker:
        label = provider if isinstance(provider, str) else provider.label
        return self._breakers[label]

    def status(self) -> list[BreakerSnapshot]:
        return [self._breakers[p.label].snapshot() for p in self.providers]

    async def complete(self, request: CompletionRequest) -> CompletionResponse:
        attempts: list[Attempt] = []
        for position, provider in enumerate(self.providers):
            breaker = self._breakers[provider.label]
            retries = 0
            while True:
                permit = breaker.acquire()
                if permit is None:
                    snap = breaker.snapshot()
                    reason = (
                        "recovery trial in progress"
                        if snap.state is BreakerState.HALF_OPEN
                        else f"circuit open, retry in {snap.retry_after:.1f}s"
                    )
                    attempt = Attempt(provider.label, "skipped", error=reason)
                    attempts.append(attempt)
                    self._log(request, provider, position, attempt)
                    break

                started = time.perf_counter()
                try:
                    response = await self._call(provider, request)
                except ProviderError as exc:
                    attempt = Attempt(
                        provider.label,
                        "error",
                        latency_ms=(time.perf_counter() - started) * 1000,
                        error=f"{type(exc).__name__}: {exc.message}",
                    )
                    attempts.append(attempt)
                    self._log(request, provider, position, attempt, trial=permit.trial)
                    if not exc.retryable:
                        permit.release()
                        raise
                    permit.failure(retry_after=exc.retry_after if isinstance(exc, RateLimitError) else None)
                    if retries < self.retry.max_retries and not (
                        isinstance(exc, RateLimitError) and exc.retry_after
                    ):
                        retries += 1
                        await asyncio.sleep(self.retry.delay(retries))
                        continue
                    break
                except BaseException:
                    # Cancellation, or a bug that is not a provider failure:
                    # give the permit back and let it propagate unmasked.
                    permit.release()
                    raise
                else:
                    permit.success()
                    attempt = Attempt(provider.label, "ok", latency_ms=response.latency_ms)
                    attempts.append(attempt)
                    self._log(request, provider, position, attempt, response=response, trial=permit.trial)
                    return replace(response, attempts=tuple(attempts))
        raise AllProvidersFailedError(self.name, attempts)

    async def _call(self, provider: Provider, request: CompletionRequest) -> CompletionResponse:
        # The provider applies the timeout inside its concurrency limit, so
        # queueing for a busy local GPU is not counted as a failure.
        return await provider.complete(request, timeout=self.attempt_timeout)

    def _log(
        self,
        request: CompletionRequest,
        provider: Provider,
        position: int,
        attempt: Attempt,
        *,
        response: CompletionResponse | None = None,
        trial: bool = False,
    ) -> None:
        if self.decision_log is None:
            return
        fields: dict[str, Any] = {
            "chain": self.name,
            "provider": provider.label,
            "model": response.model if response else provider.model,
            "position": position,
            "outcome": attempt.outcome,
            "latency_ms": round(attempt.latency_ms, 1),
        }
        if trial:
            fields["recovery_trial"] = True
        if attempt.error:
            fields["error"] = attempt.error
        if response is not None:
            fields.update(
                input_tokens=response.usage.input_tokens,
                output_tokens=response.usage.output_tokens,
                cost_usd=round(response.cost_usd, 6),
                finish_reason=response.finish_reason,
            )
        if request.metadata:
            fields["meta"] = dict(request.metadata)
        if self.log_content:
            fields["messages"] = [{"role": m.role, "content": m.content} for m in request.messages]
            if response is not None:
                fields["response"] = response.text
        self.decision_log.log("llm_call", **fields)

    async def aclose(self) -> None:
        for provider in self.providers:
            await provider.aclose()
