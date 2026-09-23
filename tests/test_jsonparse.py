from __future__ import annotations

import pytest

from tiered_llm import JSONExtractionError, parse_json


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ('{"a": 1}', {"a": 1}),
        ("[1, 2]", [1, 2]),
        ('Sure! Here you go:\n```json\n{"a": 1}\n```\nAnything else?', {"a": 1}),
        ('The result is {"a": {"b": [1, 2]}} as requested.', {"a": {"b": [1, 2]}}),
        ('<think>maybe {"a": 0}? no...</think>\n{"a": 2}', {"a": 2}),
        ("<think>ran out of tok", None),
        ('<think>draft {"a": 0} but the budget ends here', None),
        ("{'action': 'HOLD', 'ok': true, 'x': null}", {"action": "HOLD", "ok": True, "x": None}),
        ("{'flag': True, 'none': None}", {"flag": True, "none": None}),
        ("{'reason': 'claim is not true', 'ok': true}", {"reason": "claim is not true", "ok": True}),
        ("{'q': \"it's null\", 'x': null}", {"q": "it's null", "x": None}),
        ('noise {not json} then {"a": 3}', {"a": 3}),
    ],
)
def test_recovers_json(text, expected):
    if expected is None:
        with pytest.raises(JSONExtractionError):
            parse_json(text)
    else:
        assert parse_json(text) == expected


def test_scalars_are_not_accepted_as_payload():
    with pytest.raises(JSONExtractionError):
        parse_json("42")


def test_error_is_a_value_error_with_preview():
    with pytest.raises(ValueError, match="could not extract JSON"):
        parse_json("I cannot help with that.")
