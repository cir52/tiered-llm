"""Tolerant JSON extraction from model output.

Models asked for "JSON only" still wrap it in prose, Markdown fences or
``<think>`` blocks (DeepSeek-R1, Qwen3 and other reasoning models), and small
local models regularly emit Python literals (``True``, ``None``, single
quotes). This module recovers the payload in all of those cases and raises
:class:`JSONExtractionError` when there is genuinely nothing to recover.
"""

from __future__ import annotations

import ast
import json
import re
from typing import Any

__all__ = ["JSONExtractionError", "parse_json"]

_THINK_BLOCK = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_FENCE = re.compile(r"```[a-zA-Z0-9_-]*\s*(.*?)```", re.DOTALL)
_MAX_SCAN_STARTS = 64
# A quoted string (kept as is) or a bare JSON keyword (translated to Python).
_LITERAL_TOKEN = re.compile(r"""('(?:\\.|[^'\\])*'|"(?:\\.|[^"\\])*")|\b(true|false|null)\b""")
_PY_KEYWORDS = {"true": "True", "false": "False", "null": "None"}


def _python_token(match: re.Match[str]) -> str:
    return match.group(1) or _PY_KEYWORDS[match.group(2)]


class JSONExtractionError(ValueError):
    """No JSON object or array could be recovered from the text."""


def _strip_reasoning(text: str) -> str:
    text = _THINK_BLOCK.sub("", text)
    # A reasoning model that ran out of tokens leaves an unterminated block.
    lowered = text.lower()
    if "</think>" in lowered:
        text = text[lowered.rfind("</think>") + len("</think>") :]
    elif "<think>" in lowered:
        # Unterminated: everything after <think> is reasoning, never the answer.
        text = text[: lowered.find("<think>")]
    return text.strip()


def _scan(text: str) -> Any:
    decoder = json.JSONDecoder()
    starts = 0
    for index, char in enumerate(text):
        if char not in "{[":
            continue
        starts += 1
        if starts > _MAX_SCAN_STARTS:
            break
        try:
            value, _ = decoder.raw_decode(text, index)
        except ValueError:
            continue
        if isinstance(value, (dict, list)):
            return value
    raise JSONExtractionError("no JSON value found")


def _python_literal(text: str) -> Any:
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        raise JSONExtractionError("no object-like span found")
    candidate = _LITERAL_TOKEN.sub(_python_token, text[start : end + 1])
    try:
        value = ast.literal_eval(candidate)
    except (ValueError, SyntaxError, MemoryError, RecursionError) as exc:
        raise JSONExtractionError("python-literal fallback failed") from exc
    if not isinstance(value, dict):
        raise JSONExtractionError("python-literal fallback did not produce an object")
    return value


def parse_json(text: str) -> Any:
    """Return the first JSON object/array contained in ``text``.

    Order of attempts: the whole (reasoning-stripped) text, each fenced code
    block, a left-to-right scan for a decodable ``{``/``[``, and finally a
    Python-literal parse of the outermost ``{...}`` span.
    """
    cleaned = _strip_reasoning(text)
    candidates = [cleaned, *(block.strip() for block in _FENCE.findall(cleaned))]
    for candidate in candidates:
        try:
            value = json.loads(candidate)
        except ValueError:
            continue
        if isinstance(value, (dict, list)):
            return value
    for candidate in candidates:
        try:
            return _scan(candidate)
        except JSONExtractionError:
            continue
    try:
        return _python_literal(cleaned)
    except JSONExtractionError:
        pass
    preview = cleaned[:80].replace("\n", " ")
    raise JSONExtractionError(f"could not extract JSON from model output: {preview!r}")
