"""Offline providers for tests, demos and chaos drills."""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Iterable
from typing import Any, Union

from .errors import ProviderError
from .providers.base import Provider
from .types import CompletionRequest, CompletionResponse, Pricing, Usage

__all__ = ["ScriptedProvider", "Step"]

Step = Union[str, BaseException, Callable[[CompletionRequest], str]]


class ScriptedProvider(Provider):
    """Plays back a script of outcomes, one per call.

    Each step is a response text, an exception instance to raise, or a function
    of the request returning text (or raising). When the script runs out, the
    last step repeats. ``healthy`` controls :meth:`health_check`.

    >>> flaky = ScriptedProvider("flaky", [ProviderUnavailableError("503"), "ok"])
    """

    kind = "scripted"

    def __init__(
        self,
        label: str,
        script: Iterable[Step] = ("ok",),
        *,
        model: str = "scripted-model",
        latency: float = 0.0,
        usage: Usage | None = None,
        pricing: Pricing | None = None,
        healthy: bool | Callable[[], bool] = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(model=model, label=label, pricing=pricing, **kwargs)
        self.script = list(script)
        if not self.script:
            raise ValueError("script must contain at least one step")
        self.latency = latency
        self.usage = usage or Usage(10, 5)
        self.healthy = healthy
        self.calls: list[CompletionRequest] = []
        self.health_checks = 0

    async def _complete(self, request: CompletionRequest) -> CompletionResponse:
        index = min(len(self.calls), len(self.script) - 1)
        self.calls.append(request)
        step = self.script[index]
        if self.latency:
            await asyncio.sleep(self.latency)
        if isinstance(step, BaseException):
            if isinstance(step, ProviderError) and not step.provider:
                step.provider = self.label
            raise step
        text = step(request) if callable(step) else step
        return CompletionResponse(text=text, provider=self.label, model=self.model, usage=self.usage)

    async def health_check(self) -> bool:
        self.health_checks += 1
        return self.healthy() if callable(self.healthy) else self.healthy
