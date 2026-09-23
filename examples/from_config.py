"""Build a router from examples/router.toml and show its breaker status.

Needs Python 3.11+ (tomllib). Set OPENROUTER_API_KEY, ANTHROPIC_API_KEY and
GEMINI_API_KEY (any value works for this dry run - nothing is called).
"""

from __future__ import annotations

import json
from pathlib import Path

import tomllib

from tiered_llm import Router

config = tomllib.loads((Path(__file__).parent / "router.toml").read_text())
router = Router.from_config(config)

for task, chain in router.chains.items():
    print(f"{task:<7} -> " + " -> ".join(p.label for p in chain.providers))
print(json.dumps(router.status(), indent=2))
