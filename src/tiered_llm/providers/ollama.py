"""Ollama (``/api/chat``) - local models on your own hardware."""

from __future__ import annotations

import time
from collections.abc import Mapping, Sequence
from typing import Any

from ..errors import ProviderError, ResponseParseError
from ..types import CompletionRequest, CompletionResponse, Usage
from .base import HTTPProvider, with_json_instruction

__all__ = ["OllamaProvider", "choose_model"]


def choose_model(preferences: Sequence[str], installed: Sequence[str]) -> str | None:
    """Pick the first preferred model that is actually pulled.

    Exact (case-insensitive) matches win over prefix matches, so ``"gemma3"``
    resolves to ``"gemma3:12b"`` only if no model is literally called ``gemma3``.
    """
    by_lower = {name.lower(): name for name in installed}
    for pref in preferences:
        if pref.lower() in by_lower:
            return by_lower[pref.lower()]
    for pref in preferences:
        for lower, real in by_lower.items():
            if lower.startswith(pref.lower()):
                return real
    return None


class OllamaProvider(HTTPProvider):
    """Local inference through Ollama.

    ``max_concurrency`` defaults to 1: parallel requests to one consumer GPU
    make Ollama load several runners that then fight over VRAM, which is slower
    than queueing. Raise it if you run ``OLLAMA_NUM_PARALLEL`` or several GPUs.
    """

    kind = "ollama"

    def __init__(
        self,
        model: str,
        *,
        base_url: str = "http://localhost:11434",
        timeout: float = 120.0,
        keep_alive: str | None = "60m",
        options: Mapping[str, Any] | None = None,
        max_concurrency: int | None = 1,
        models_cache_ttl: float = 300.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            model=model,
            base_url=base_url,
            timeout=timeout,
            max_concurrency=max_concurrency,
            **kwargs,
        )
        self.keep_alive = keep_alive
        self.options = dict(options or {})
        self._models_cache_ttl = models_cache_ttl
        self._models_cache: tuple[float, list[str]] | None = None

    def build_payload(self, request: CompletionRequest) -> dict[str, Any]:
        messages: list[dict[str, str]] = []
        system = with_json_instruction(request.system_prompt, request.json_mode)
        if system:
            messages.append({"role": "system", "content": system})
        messages.extend({"role": m.role, "content": m.content} for m in request.conversation)
        options: dict[str, Any] = {**self.options, "num_predict": request.max_tokens}
        if request.temperature is not None:
            options["temperature"] = request.temperature
        if request.stop:
            options["stop"] = list(request.stop)
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": options,
        }
        if request.json_mode:
            payload["format"] = "json"
        if self.keep_alive is not None:
            payload["keep_alive"] = self.keep_alive
        return payload

    async def _complete(self, request: CompletionRequest) -> CompletionResponse:
        data = await self._post_json("/api/chat", self.build_payload(request))
        if data.get("error"):
            raise ResponseParseError(str(data["error"]), provider=self.label)
        message = data.get("message")
        if not isinstance(message, dict):
            raise ResponseParseError("missing 'message' in response", provider=self.label)
        return CompletionResponse(
            text=str(message.get("content") or ""),
            provider=self.label,
            model=str(data.get("model") or self.model),
            usage=Usage(int(data.get("prompt_eval_count", 0)), int(data.get("eval_count", 0))),
            finish_reason=data.get("done_reason"),
            raw=data,
        )

    async def installed_models(self, *, refresh: bool = False) -> list[str]:
        """Names from ``/api/tags``, cached for ``models_cache_ttl`` seconds."""
        now = time.monotonic()
        if (
            not refresh
            and self._models_cache is not None
            and now - self._models_cache[0] < self._models_cache_ttl
        ):
            return list(self._models_cache[1])
        response = await self._request("GET", "/api/tags", timeout=5.0)
        try:
            models = response.json().get("models", [])
        except (ValueError, AttributeError) as exc:
            raise ResponseParseError("unexpected /api/tags response", provider=self.label) from exc
        names = [str(m["name"]) for m in models if isinstance(m, dict) and m.get("name")]
        self._models_cache = (now, names)
        return list(names)

    async def health_check(self) -> bool:
        """Healthy when the server answers and the configured model is pulled.

        Costs no GPU time, unlike a test completion.
        """
        try:
            names = await self.installed_models(refresh=True)
        except ProviderError:
            return False
        # Match the way /api/chat resolves names: an untagged model means ":latest".
        wanted = self.model if ":" in self.model else f"{self.model}:latest"
        return wanted.lower() in {name.lower() for name in names}
