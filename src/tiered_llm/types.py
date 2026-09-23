"""Provider-neutral request and response types."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any, Literal

from .jsonparse import parse_json

__all__ = [
    "FREE",
    "Attempt",
    "AttemptOutcome",
    "CompletionRequest",
    "CompletionResponse",
    "Message",
    "Pricing",
    "Role",
    "Usage",
]

Role = Literal["system", "user", "assistant"]
AttemptOutcome = Literal["ok", "error", "skipped"]


@dataclass(frozen=True, slots=True)
class Message:
    role: Role
    content: str

    @classmethod
    def system(cls, content: str) -> Message:
        return cls("system", content)

    @classmethod
    def user(cls, content: str) -> Message:
        return cls("user", content)

    @classmethod
    def assistant(cls, content: str) -> Message:
        return cls("assistant", content)


@dataclass(frozen=True, slots=True)
class CompletionRequest:
    """One chat completion, expressed once and translated by every provider.

    ``metadata`` is never sent to a model. It travels into the decision log,
    so put correlation ids there (``request_id``, ``item_id``, ``stage`` ...).
    """

    messages: tuple[Message, ...]
    max_tokens: int = 1024
    temperature: float | None = None
    json_mode: bool = False
    stop: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # Accept lists for convenience but store tuples so requests stay immutable.
        object.__setattr__(self, "messages", tuple(self.messages))
        object.__setattr__(self, "stop", tuple(self.stop))
        if not self.messages:
            raise ValueError("CompletionRequest needs at least one message")
        if not any(m.role != "system" for m in self.messages):
            raise ValueError("CompletionRequest needs at least one user or assistant message")
        if self.max_tokens < 1:
            raise ValueError("max_tokens must be >= 1")

    @classmethod
    def from_prompt(
        cls,
        prompt: str,
        *,
        system: str | None = None,
        max_tokens: int = 1024,
        temperature: float | None = None,
        json_mode: bool = False,
        stop: Iterable[str] = (),
        metadata: Mapping[str, Any] | None = None,
    ) -> CompletionRequest:
        messages = [Message.system(system)] if system else []
        messages.append(Message.user(prompt))
        return cls(
            messages=tuple(messages),
            max_tokens=max_tokens,
            temperature=temperature,
            json_mode=json_mode,
            stop=tuple(stop),
            metadata=dict(metadata or {}),
        )

    def with_metadata(self, **values: Any) -> CompletionRequest:
        """Return a copy with extra metadata merged in."""
        return CompletionRequest(
            messages=self.messages,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            json_mode=self.json_mode,
            stop=self.stop,
            metadata={**self.metadata, **values},
        )

    @property
    def system_prompt(self) -> str | None:
        parts = [m.content for m in self.messages if m.role == "system"]
        return "\n\n".join(parts) if parts else None

    @property
    def conversation(self) -> tuple[Message, ...]:
        return tuple(m for m in self.messages if m.role != "system")


@dataclass(frozen=True, slots=True)
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


@dataclass(frozen=True, slots=True)
class Pricing:
    """USD per one million tokens. Prices change; keep them in config, not code."""

    input_per_mtok: float = 0.0
    output_per_mtok: float = 0.0

    def cost(self, usage: Usage) -> float:
        return (
            usage.input_tokens * self.input_per_mtok + usage.output_tokens * self.output_per_mtok
        ) / 1_000_000


FREE = Pricing()


@dataclass(frozen=True, slots=True)
class Attempt:
    """One step of a fallback chain, as recorded on the response and in the log."""

    provider: str
    outcome: AttemptOutcome
    latency_ms: float = 0.0
    error: str | None = None


@dataclass(frozen=True, slots=True)
class CompletionResponse:
    text: str
    provider: str
    model: str
    usage: Usage = field(default_factory=Usage)
    latency_ms: float = 0.0
    cost_usd: float = 0.0
    finish_reason: str | None = None
    attempts: tuple[Attempt, ...] = ()
    raw: Any = field(default=None, repr=False, compare=False)

    @property
    def fell_back(self) -> bool:
        """True if at least one provider before the answering one failed or was skipped."""
        return any(a.outcome != "ok" for a in self.attempts)

    def parse_json(self) -> Any:
        """Extract JSON from ``text`` (tolerates fences, prose and ``<think>`` blocks)."""
        return parse_json(self.text)
