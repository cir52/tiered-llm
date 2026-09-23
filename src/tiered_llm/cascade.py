"""Two-stage routing: a cheap model screens everything, an expensive one decides the rest.

This is the split that made Trade52's economics work: a local model on owned
hardware assessed every candidate, and only the survivors reached a frontier
model. The same shape fits document triage, ticket routing, lead
qualification or contract review - any workload where volume is high and only
a few items deserve the expensive model.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Iterable, Sequence
from dataclasses import dataclass
from typing import Literal, Protocol, overload

from .decision_log import DecisionLog
from .types import CompletionRequest, CompletionResponse

__all__ = [
    "Cascade",
    "CascadeItem",
    "CascadeResult",
    "CascadeStats",
    "Completer",
    "Gate",
]


class Completer(Protocol):
    """Anything that completes a request: a Provider, a FallbackChain ..."""

    def complete(self, request: CompletionRequest) -> Awaitable[CompletionResponse]: ...


Gate = Callable[[CompletionResponse], bool]
DecideRequest = CompletionRequest | Callable[[CompletionResponse], CompletionRequest]
GateErrorPolicy = Literal["reject", "escalate", "raise"]


@dataclass(frozen=True, slots=True)
class CascadeItem:
    screen: CompletionRequest
    decide: DecideRequest
    """A fixed request, or a function building it from the screening response."""
    item_id: str | None = None


@dataclass(frozen=True, slots=True)
class CascadeResult:
    item_id: str | None
    escalated: bool
    screen: CompletionResponse
    decision: CompletionResponse | None = None
    gate_error: str | None = None

    @property
    def final(self) -> CompletionResponse:
        return self.decision if self.decision is not None else self.screen

    @property
    def cost_usd(self) -> float:
        return self.screen.cost_usd + (self.decision.cost_usd if self.decision else 0.0)


@dataclass(frozen=True, slots=True)
class CascadeStats:
    items: int
    escalated: int
    screen_cost_usd: float
    decide_cost_usd: float

    @property
    def escalation_rate(self) -> float:
        return self.escalated / self.items if self.items else 0.0

    @property
    def total_cost_usd(self) -> float:
        return self.screen_cost_usd + self.decide_cost_usd

    @classmethod
    def from_results(cls, results: Iterable[CascadeResult]) -> CascadeStats:
        items = escalated = 0
        screen_cost = decide_cost = 0.0
        for result in results:
            items += 1
            screen_cost += result.screen.cost_usd
            if result.decision is not None:
                escalated += 1
                decide_cost += result.decision.cost_usd
        return cls(items, escalated, screen_cost, decide_cost)


class Cascade:
    def __init__(
        self,
        *,
        screen: Completer,
        decide: Completer,
        gate: Gate,
        on_gate_error: GateErrorPolicy = "raise",
        decision_log: DecisionLog | None = None,
        name: str = "cascade",
    ) -> None:
        """
        :param gate: returns True if the screening response warrants the expensive
            model. It usually parses the screener's JSON: ``lambda r: r.parse_json()["score"] >= 0.7``.
        :param on_gate_error: what to do when the gate raises (e.g. the small model
            returned unparseable output): ``"reject"`` (fail closed, cheapest),
            ``"escalate"`` (fail open, let the big model look), or ``"raise"``.
        """
        if on_gate_error not in ("reject", "escalate", "raise"):
            raise ValueError("on_gate_error must be 'reject', 'escalate' or 'raise'")
        self.screen = screen
        self.decide = decide
        self.gate = gate
        self.on_gate_error = on_gate_error
        self.decision_log = decision_log
        self.name = name

    async def run(
        self,
        screen: CompletionRequest,
        decide: DecideRequest,
        *,
        item_id: str | None = None,
    ) -> CascadeResult:
        meta = {"cascade": self.name, **({"item_id": item_id} if item_id is not None else {})}
        screened = await self.screen.complete(screen.with_metadata(stage="screen", **meta))

        gate_error: str | None = None
        try:
            escalate = bool(self.gate(screened))
        except Exception as exc:
            if self.on_gate_error == "raise":
                raise
            gate_error = f"{type(exc).__name__}: {exc}"
            escalate = self.on_gate_error == "escalate"

        decision: CompletionResponse | None = None
        if escalate:
            request = decide(screened) if callable(decide) else decide
            decision = await self.decide.complete(request.with_metadata(stage="decide", **meta))

        result = CascadeResult(item_id, escalate, screened, decision, gate_error)
        if self.decision_log is not None:
            self.decision_log.log(
                "cascade_verdict",
                cascade=self.name,
                item_id=item_id,
                escalated=escalate,
                gate_error=gate_error,
                screen_provider=screened.provider,
                decide_provider=decision.provider if decision else None,
                cost_usd=round(result.cost_usd, 6),
            )
        return result

    @overload
    async def run_many(
        self,
        items: Sequence[CascadeItem],
        *,
        concurrency: int = ...,
        return_exceptions: Literal[False] = ...,
    ) -> list[CascadeResult]: ...

    @overload
    async def run_many(
        self,
        items: Sequence[CascadeItem],
        *,
        concurrency: int = ...,
        return_exceptions: Literal[True],
    ) -> list[CascadeResult | BaseException]: ...

    async def run_many(
        self,
        items: Sequence[CascadeItem],
        *,
        concurrency: int = 8,
        return_exceptions: bool = False,
    ) -> list[CascadeResult] | list[CascadeResult | BaseException]:
        """Run many items with bounded concurrency; results keep input order.

        Without ``return_exceptions`` the first failure cancels the remaining
        items, so no paid calls keep running for a result nobody will read.
        """
        semaphore = asyncio.Semaphore(max(1, concurrency))

        async def one(item: CascadeItem) -> CascadeResult:
            async with semaphore:
                return await self.run(item.screen, item.decide, item_id=item.item_id)

        tasks = [asyncio.ensure_future(one(item)) for item in items]
        if not tasks:
            return []
        if return_exceptions:
            return list(await asyncio.gather(*tasks, return_exceptions=True))
        try:
            _, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_EXCEPTION)
        except BaseException:  # the caller cancelled us
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            raise
        for task in pending:
            task.cancel()
        await asyncio.gather(*pending, return_exceptions=True)
        for task in tasks:
            if not task.cancelled() and task.exception() is not None:
                raise task.exception()  # type: ignore[misc]
        return [task.result() for task in tasks]
