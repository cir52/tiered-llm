"""Anthropic Messages API."""

from __future__ import annotations

from typing import Any

import httpx

from ..errors import ProviderError, ProviderUnavailableError, ResponseParseError
from ..types import CompletionRequest, CompletionResponse, Usage
from .base import HTTPProvider, resolve_api_key, with_json_instruction

__all__ = ["AnthropicProvider"]


class AnthropicProvider(HTTPProvider):
    kind = "anthropic"

    def __init__(
        self,
        model: str,
        *,
        api_key: str | None = None,
        api_key_env: str = "ANTHROPIC_API_KEY",
        base_url: str = "https://api.anthropic.com",
        api_version: str = "2023-06-01",
        **kwargs: Any,
    ) -> None:
        super().__init__(model=model, base_url=base_url, **kwargs)
        self._api_key = resolve_api_key(self.kind, api_key, api_key_env)
        self.api_version = api_version

    def _auth_headers(self) -> dict[str, str]:
        return {"x-api-key": self._api_key, "anthropic-version": self.api_version}

    def _classify_error(self, response: httpx.Response, message: str) -> ProviderError:
        # Anthropic reports an exhausted credit balance as a 400. That is a
        # provider problem (fall back), not a malformed request.
        if response.status_code == 400 and "credit balance" in message.lower():
            return ProviderUnavailableError(message, provider=self.label, status_code=400)
        return super()._classify_error(response, message)

    def build_payload(self, request: CompletionRequest) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self.model,
            "max_tokens": request.max_tokens,
            "messages": [{"role": m.role, "content": m.content} for m in request.conversation],
        }
        system = with_json_instruction(request.system_prompt, request.json_mode)
        if system:
            payload["system"] = system
        if request.temperature is not None:
            payload["temperature"] = request.temperature
        if request.stop:
            payload["stop_sequences"] = list(request.stop)
        return payload

    async def _complete(self, request: CompletionRequest) -> CompletionResponse:
        data = await self._post_json("/v1/messages", self.build_payload(request))
        blocks = data.get("content")
        if not isinstance(blocks, list):
            raise ResponseParseError("missing 'content' in response", provider=self.label)
        text = "".join(
            str(block.get("text", ""))
            for block in blocks
            if isinstance(block, dict) and block.get("type") == "text"
        )
        usage = data.get("usage") or {}
        return CompletionResponse(
            text=text,
            provider=self.label,
            model=str(data.get("model") or self.model),
            usage=Usage(int(usage.get("input_tokens", 0)), int(usage.get("output_tokens", 0))),
            finish_reason=data.get("stop_reason"),
            raw=data,
        )
