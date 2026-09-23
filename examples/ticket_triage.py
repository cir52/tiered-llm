"""Two-stage triage: a local model screens every ticket, a frontier model
only answers the ones that need it.

Offline by default (scripted models, so it runs anywhere). With --live it
uses a real Ollama model for screening and Anthropic for the final answer:

    python examples/ticket_triage.py
    OLLAMA_MODEL=qwen3:8b ANTHROPIC_API_KEY=... python examples/ticket_triage.py --live
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os

from tiered_llm import (
    AnthropicProvider,
    Cascade,
    CascadeItem,
    CascadeStats,
    CompletionRequest,
    CompletionResponse,
    FallbackChain,
    OllamaProvider,
    Pricing,
    Usage,
)
from tiered_llm.testing import ScriptedProvider

TICKETS = {
    "T-101": "How do I change the email address on my account?",
    "T-102": "Your last invoice charged us twice and finance is asking. Please fix today.",
    "T-103": "Love the new dashboard, thanks!",
    "T-104": "Production API returns 500 on every request since 09:00, our checkout is down.",
    "T-105": "Where can I download last month's invoice?",
    "T-106": "Is there a dark mode?",
    "T-107": "We are evaluating you for 400 seats, need a DPA and SSO details before Friday.",
    "T-108": "Password reset email never arrives.",
    "T-109": "Can I export my data to CSV?",
    "T-110": "Our data was visible to another customer in the shared report view. Urgent.",
}

SCREEN_SYSTEM = (
    "You triage support tickets. Return JSON: "
    '{"escalate": true|false, "category": "billing|outage|security|sales|howto|feedback", '
    '"reason": "<10 words"}. Escalate outages, security, money problems and large sales leads; '
    "everything answerable from the help center is not escalated."
)
DECIDE_SYSTEM = "You are a senior support engineer. Draft a precise first reply (max 80 words)."

# Illustrative list prices, USD per million tokens. Keep real prices in config.
FRONTIER_PRICING = Pricing(input_per_mtok=3.0, output_per_mtok=15.0)


def scripted_models() -> tuple[ScriptedProvider, ScriptedProvider]:
    urgent = ("twice", "500", "seats", "visible to another")

    def screen(request: CompletionRequest) -> str:
        text = request.messages[-1].content
        escalate = any(word in text for word in urgent)
        return json.dumps({"escalate": escalate, "category": "outage" if "500" in text else "other"})

    local = ScriptedProvider("ollama:local", [screen], usage=Usage(180, 25))
    frontier = ScriptedProvider(
        "anthropic:frontier",
        [lambda r: "Thanks for flagging this - we're on it. ..."],
        usage=Usage(420, 110),
        pricing=FRONTIER_PRICING,
    )
    return local, frontier


def live_models() -> tuple[OllamaProvider, AnthropicProvider]:
    local = OllamaProvider(os.environ.get("OLLAMA_MODEL", "qwen3:8b"))
    frontier = AnthropicProvider(
        os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-5"), pricing=FRONTIER_PRICING
    )
    return local, frontier


def should_escalate(screened: CompletionResponse) -> bool:
    return bool(screened.parse_json().get("escalate"))


async def main(live: bool) -> None:
    local, frontier = live_models() if live else scripted_models()
    screen_chain = FallbackChain([local], name="screen")
    decide_chain = FallbackChain([frontier], name="decide")
    cascade = Cascade(
        screen=screen_chain,
        decide=decide_chain,
        gate=should_escalate,
        on_gate_error="escalate",  # if the small model babbles, let the big one look
    )

    def decide_request(ticket: str):
        def build(screened: CompletionResponse) -> CompletionRequest:
            return CompletionRequest.from_prompt(
                f"Ticket:\n{ticket}\n\nTriage notes: {screened.text}", system=DECIDE_SYSTEM, max_tokens=300
            )

        return build

    items = [
        CascadeItem(
            screen=CompletionRequest.from_prompt(text, system=SCREEN_SYSTEM, json_mode=True, max_tokens=120),
            decide=decide_request(text),
            item_id=ticket_id,
        )
        for ticket_id, text in TICKETS.items()
    ]
    results = await cascade.run_many(items, concurrency=4)

    for result in results:
        mark = "ESCALATED" if result.escalated else "self-serve"
        print(f"{result.item_id}  {mark:<10}  {TICKETS[result.item_id or ''][:62]}")

    stats = CascadeStats.from_results(results)
    per_call = next((r.decision.cost_usd for r in results if r.decision), 0.0)
    baseline = per_call * stats.items
    print(
        f"\n{stats.escalated}/{stats.items} tickets reached the frontier model ({stats.escalation_rate:.0%})"
    )
    print(f"cost: ${stats.total_cost_usd:.4f}  vs  ${baseline:.4f} sending everything to the frontier model")
    await screen_chain.aclose()
    await decide_chain.aclose()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--live", action="store_true", help="use Ollama + Anthropic instead of scripted models"
    )
    asyncio.run(main(parser.parse_args().live))
