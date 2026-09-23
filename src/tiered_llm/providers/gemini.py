"""Google Gemini (Generative Language API, ``generateContent``)."""

from __future__ import annotations

from typing import Any
from urllib.parse import quote

import httpx

from ..errors import AuthenticationError, ProviderError, ResponseParseError
from ..types import CompletionRequest, CompletionResponse, Usage
from .base import HTTPProvider, resolve_api_key

__all__ = ["GeminiProvider"]

_BLOCKED_FINISH_REASONS = frozenset(
    {"SAFETY", "RECITATION", "BLOCKLIST", "PROHIBITED_CONTENT", "SPII", "IMAGE_SAFETY", "OTHER"}
)


class GeminiProvider(HTTPProvider):
    kind = "gemini"

    def __init__(
        self,
        model: str,
        *,
        api_key: str | None = None,
        api_key_env: str = "GEMINI_API_KEY",
        base_url: str = "https://generativelanguage.googleapis.com",
        api_version: str = "v1beta",
        **kwargs: Any,
    ) -> None:
        super().__init__(model=model, base_url=base_url, **kwargs)
        self._api_key = resolve_api_key(self.kind, api_key, api_key_env)
        self.api_version = api_version

    def _auth_headers(self) -> dict[str, str]:
        return {"x-goog-api-key": self._api_key}

    def _classify_error(self, response: httpx.Response, message: str) -> ProviderError:
        # Gemini reports a bad API key as 400 INVALID_ARGUMENT / API_KEY_INVALID.
        if response.status_code == 400 and (
            "API_KEY_INVALID" in response.text or "api key not valid" in message.lower()
        ):
            return AuthenticationError(message, provider=self.label, status_code=400)
        return super()._classify_error(response, message)

    def build_payload(self, request: CompletionRequest) -> dict[str, Any]:
        contents = [
            {"role": "model" if m.role == "assistant" else "user", "parts": [{"text": m.content}]}
            for m in request.conversation
        ]
        config: dict[str, Any] = {"maxOutputTokens": request.max_tokens}
        if request.temperature is not None:
            config["temperature"] = request.temperature
        if request.stop:
            config["stopSequences"] = list(request.stop)
        if request.json_mode:
            config["responseMimeType"] = "application/json"
        payload: dict[str, Any] = {"contents": contents, "generationConfig": config}
        if request.system_prompt:
            payload["systemInstruction"] = {"parts": [{"text": request.system_prompt}]}
        return payload

    async def _complete(self, request: CompletionRequest) -> CompletionResponse:
        path = f"/{self.api_version}/models/{quote(self.model, safe='')}:generateContent"
        data = await self._post_json(path, self.build_payload(request))
        candidates = data.get("candidates") or []
        if not candidates:
            reason = (data.get("promptFeedback") or {}).get("blockReason", "no candidates")
            raise ResponseParseError(f"empty response ({reason})", provider=self.label)
        candidate = candidates[0]
        parts = (candidate.get("content") or {}).get("parts") or []
        # Thinking models return their reasoning as parts flagged "thought".
        text = "".join(str(p.get("text", "")) for p in parts if isinstance(p, dict) and not p.get("thought"))
        finish_reason = candidate.get("finishReason")
        if not text and finish_reason in _BLOCKED_FINISH_REASONS:
            raise ResponseParseError(f"candidate blocked ({finish_reason})", provider=self.label)
        meta = data.get("usageMetadata") or {}
        output_tokens = int(meta.get("candidatesTokenCount", 0)) + int(meta.get("thoughtsTokenCount", 0))
        return CompletionResponse(
            text=text,
            provider=self.label,
            model=str(data.get("modelVersion") or self.model),
            usage=Usage(int(meta.get("promptTokenCount", 0)), output_tokens),
            finish_reason=finish_reason,
            raw=data,
        )
