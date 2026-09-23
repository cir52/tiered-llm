"""Per-task model selection over shared circuit breakers.

A router maps task names ("screen", "decide", "summarize" ...) to fallback
chains. All chains share one breaker per provider, so an outage discovered by
one task is immediately respected by every other task using that provider.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from .breaker import BreakerConfig, BreakerRegistry
from .chain import FallbackChain, RetryPolicy, log_breaker_transitions
from .decision_log import DecisionLog
from .health import HealthMonitor
from .providers import build_provider
from .providers.base import Provider
from .types import CompletionRequest, CompletionResponse

__all__ = ["Router"]


class Router:
    def __init__(
        self,
        routes: Mapping[str, Sequence[Provider]],
        *,
        breaker_config: BreakerConfig | None = None,
        retry: RetryPolicy | None = None,
        attempt_timeout: float | None = None,
        decision_log: DecisionLog | None = None,
        log_content: bool = False,
    ) -> None:
        if not routes:
            raise ValueError("a router needs at least one route")
        self.providers: dict[str, Provider] = {}
        for chain_providers in routes.values():
            for provider in chain_providers:
                known = self.providers.setdefault(provider.label, provider)
                if known is not provider:
                    raise ValueError(f"two different provider objects share the label '{provider.label}'")
        self.decision_log = decision_log
        self.registry = BreakerRegistry(breaker_config)
        if decision_log is not None:
            log_breaker_transitions(self.registry, decision_log)
        self.chains: dict[str, FallbackChain] = {
            task: FallbackChain(
                chain_providers,
                name=task,
                registry=self.registry,
                retry=retry,
                attempt_timeout=attempt_timeout,
                decision_log=decision_log,
                log_content=log_content,
            )
            for task, chain_providers in routes.items()
        }

    @classmethod
    def from_config(
        cls,
        config: Mapping[str, Any],
        *,
        decision_log: DecisionLog | None = None,
    ) -> Router:
        """Build providers and routes from one plain mapping (e.g. parsed TOML/YAML/JSON).

        Swapping a vendor, reordering a fallback chain or moving a task to a
        cheaper model becomes a config change instead of a code change::

            {
              "breaker": {"failure_threshold": 3, "recovery_timeout": 30},
              "providers": {
                "local":  {"type": "ollama", "model": "qwen3:14b"},
                "claude": {"type": "anthropic", "model": "claude-sonnet-4-5",
                           "pricing": {"input_per_mtok": 3, "output_per_mtok": 15}},
              },
              "routes": {"screen": ["local"], "decide": ["claude", "local"]},
            }
        """
        providers_cfg: Mapping[str, Mapping[str, Any]] = config.get("providers") or {}
        routes_cfg: Mapping[str, Sequence[str]] = config.get("routes") or {}
        providers = {label: build_provider(label, spec) for label, spec in providers_cfg.items()}
        routes: dict[str, list[Provider]] = {}
        for task, labels in routes_cfg.items():
            missing = [label for label in labels if label not in providers]
            if missing:
                raise ValueError(f"route '{task}' references unknown providers: {missing}")
            routes[task] = [providers[label] for label in labels]
        breaker_cfg = config.get("breaker")
        retry_cfg = config.get("retry")
        return cls(
            routes,
            breaker_config=BreakerConfig(**breaker_cfg) if breaker_cfg else None,
            retry=RetryPolicy(**retry_cfg) if retry_cfg else None,
            attempt_timeout=config.get("attempt_timeout"),
            decision_log=decision_log,
            log_content=bool(config.get("log_content", False)),
        )

    def chain(self, task: str) -> FallbackChain:
        try:
            return self.chains[task]
        except KeyError:
            raise KeyError(f"no route for task '{task}' (known: {sorted(self.chains)})") from None

    async def complete(self, task: str, request: CompletionRequest) -> CompletionResponse:
        return await self.chain(task).complete(request.with_metadata(task=task))

    def status(self) -> dict[str, dict[str, object]]:
        """Breaker state per provider - ready to return from a /health endpoint."""
        return {snap.name: snap.as_dict() for snap in self.registry.snapshots()}

    def health_monitor(self, *, interval: float = 15.0, probe_timeout: float = 12.0) -> HealthMonitor:
        targets = [(p, self.registry.get(label)) for label, p in self.providers.items()]
        return HealthMonitor(
            targets, interval=interval, probe_timeout=probe_timeout, decision_log=self.decision_log
        )

    async def aclose(self) -> None:
        for provider in self.providers.values():
            await provider.aclose()
