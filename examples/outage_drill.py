"""A simulated 3 a.m. outage, end to end, offline.

The primary provider starts failing mid-stream. Watch the chain fall back,
the circuit open (so no more requests are wasted on the dead provider), the
health monitor notice the recovery, and traffic return to the primary.
Every step lands in the decision log, summarised at the end.

    python examples/outage_drill.py
"""

from __future__ import annotations

import asyncio
import collections
import tempfile
from pathlib import Path

from tiered_llm import (
    BreakerConfig,
    CompletionRequest,
    DecisionLog,
    ProviderUnavailableError,
    Router,
    iter_decisions,
)
from tiered_llm.testing import ScriptedProvider


class Upstream:
    """A fake vendor we can switch off and on."""

    def __init__(self) -> None:
        self.down = False

    def answer(self, request: CompletionRequest) -> str:
        if self.down:
            raise ProviderUnavailableError("503 Service Unavailable")
        return "primary answer"


async def main() -> None:
    upstream = Upstream()
    primary = ScriptedProvider("claude", [upstream.answer], healthy=lambda: not upstream.down)
    backup = ScriptedProvider("gemini", ["backup answer"])

    log_path = Path(tempfile.mkdtemp()) / "decisions.jsonl"
    log = DecisionLog(log_path, fsync=False)
    router = Router(
        {"decide": [primary, backup]},
        # Short timeouts so the drill runs in a few seconds; defaults are 3 / 30s / 180s.
        breaker_config=BreakerConfig(failure_threshold=3, recovery_timeout=0.6, max_recovery_timeout=2),
        decision_log=log,
    )

    print("req  route                       circuit(claude)")
    async with router.health_monitor(interval=0.2, probe_timeout=1):
        for n in range(1, 31):
            if n == 6:
                upstream.down = True
                print("---  claude goes down  ---")
            if n == 20:
                upstream.down = False
                print("---  claude recovers   ---")

            response = await router.complete("decide", CompletionRequest.from_prompt(f"request {n}"))
            route = " -> ".join(
                f"{a.provider} {'ok' if a.outcome == 'ok' else 'SKIP' if a.outcome == 'skipped' else 'FAIL'}"
                for a in response.attempts
            )
            state = router.status()["claude"]["state"]
            print(f"#{n:02d}  {route:<27} {state}")
            await asyncio.sleep(0.1)

    log.close()
    events = list(iter_decisions(log_path))
    calls = collections.Counter((e["provider"], e["outcome"]) for e in events if e["event"] == "llm_call")
    print("\nDecision log summary")
    for (provider, outcome), count in sorted(calls.items()):
        print(f"  {provider:<7} {outcome:<8} {count}")
    for e in events:
        if e["event"] == "breaker_transition":
            print(f"  {e['ts'][11:23]}  circuit {e['provider']}: {e['old']} -> {e['new']}")
    print(f"\nclaude was actually called {len(primary.calls)} times for 30 requests;")
    print("while its circuit was open, requests went straight to gemini without waiting on a dead API.")
    print(f"full log: {log_path}")


if __name__ == "__main__":
    asyncio.run(main())
