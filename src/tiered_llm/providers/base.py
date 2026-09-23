"""Provider interface and the shared HTTP plumbing."""

from __future__ import annotations

import asyncio
import os
import threading
import time
import weakref
from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import replace
from email.utils import parsedate_to_datetime
from typing import Any, ClassVar

import httpx

from ..breaker import BreakerConfig
from ..concurrency import ConcurrencyLimiter
from ..errors import (
    ProviderError,
    ProviderUnavailableError,
    ResponseParseError,
    error_for_status,
)
from ..types import FREE, CompletionRequest, CompletionResponse, Pricing

__all__ = ["JSON_INSTRUCTION", "HTTPProvider", "Provider", "resolve_api_key"]

JSON_INSTRUCTION = "Respond with a single valid JSON value and nothing else."


def resolve_api_key(kind: str, api_key: str | None, env_var: str | None) -> str:
    key = api_key or (os.environ.get(env_var) if env_var else None)
    if not key:
        hint = f" or set ${env_var}" if env_var else ""
        raise ValueError(f"{kind}: no API key - pass api_key={hint}")
    return key


def with_json_instruction(system: str | None, json_mode: bool) -> str | None:
    if not json_mode:
        return system
    return f"{system}\n\n{JSON_INSTRUCTION}" if system else JSON_INSTRUCTION


class Provider(ABC):
    """One model behind one endpoint.

    Subclasses implement :meth:`_complete` and translate the neutral request
    into their wire format. Everything cross-cutting (concurrency limit,
    latency, cost) happens here so that every provider behaves the same way.
    """

    kind: ClassVar[str] = "provider"

    def __init__(
        self,
        *,
        model: str,
        label: str | None = None,
        pricing: Pricing | None = None,
        max_concurrency: int | None = None,
        breaker_config: BreakerConfig | None = None,
    ) -> None:
        self.model = model
        self.label = label or f"{self.kind}:{model}"
        self.pricing = pricing or FREE
        self.breaker_config = breaker_config
        """Optional per-provider breaker settings (e.g. a flakier API gets a lower threshold)."""
        self._limiter = ConcurrencyLimiter(max_concurrency) if max_concurrency else None

    def __repr__(self) -> str:
        return f"{type(self).__name__}(label={self.label!r}, model={self.model!r})"

    async def complete(
        self, request: CompletionRequest, *, timeout: float | None = None
    ) -> CompletionResponse:
        """Run one completion.

        ``timeout`` bounds the call itself. Time spent queueing for a
        ``max_concurrency`` slot does not count, so a busy local GPU is not
        mistaken for a broken one.
        """
        if self._limiter is None:
            return await self._timed(request, timeout)
        async with self._limiter:
            return await self._timed(request, timeout)

    async def _timed(self, request: CompletionRequest, timeout: float | None) -> CompletionResponse:
        started = time.perf_counter()
        if timeout is None:
            response = await self._complete(request)
        else:
            try:
                response = await asyncio.wait_for(self._complete(request), timeout)
            except asyncio.TimeoutError as exc:
                raise ProviderUnavailableError(
                    f"attempt timed out after {timeout:g}s", provider=self.label
                ) from exc
        return replace(
            response,
            latency_ms=(time.perf_counter() - started) * 1000,
            cost_usd=self.pricing.cost(response.usage),
        )

    @abstractmethod
    async def _complete(self, request: CompletionRequest) -> CompletionResponse:
        """Perform the call. Raise a :class:`~tiered_llm.errors.ProviderError` on failure."""

    async def health_check(self) -> bool:
        """Cheap liveness probe used while the circuit is open.

        The default sends a one-token completion. Local servers override this
        with a free status endpoint.
        """
        try:
            await self.complete(CompletionRequest.from_prompt("ping", max_tokens=1))
        except ProviderError:
            return False
        return True

    async def aclose(self) -> None:  # noqa: B027 - optional hook, not abstract
        """Release network resources."""


def _parse_retry_after(value: str | None) -> float | None:
    if not value:
        return None
    try:
        return max(0.0, float(value))
    except ValueError:
        pass
    try:
        return max(0.0, parsedate_to_datetime(value).timestamp() - time.time())
    except (TypeError, ValueError, IndexError, OverflowError):
        return None


def _error_message(response: httpx.Response) -> str:
    try:
        data = response.json()
    except ValueError:
        return response.text[:300] or response.reason_phrase
    if isinstance(data, dict):
        error = data.get("error")
        if isinstance(error, dict) and error.get("message"):
            return str(error["message"])[:300]
        if isinstance(error, str):
            return error[:300]
        if data.get("message"):
            return str(data["message"])[:300]
    return str(data)[:300]


class HTTPProvider(Provider):
    """Base for providers that speak JSON over HTTP."""

    def __init__(
        self,
        *,
        model: str,
        base_url: str,
        timeout: float = 45.0,
        headers: Mapping[str, str] | None = None,
        client: httpx.AsyncClient | None = None,
        label: str | None = None,
        pricing: Pricing | None = None,
        max_concurrency: int | None = None,
        breaker_config: BreakerConfig | None = None,
    ) -> None:
        super().__init__(
            model=model,
            label=label,
            pricing=pricing,
            max_concurrency=max_concurrency,
            breaker_config=breaker_config,
        )
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._extra_headers = dict(headers or {})
        self._client = client
        # httpx connection pools belong to the loop that opened them. Owned
        # clients are therefore kept per loop, so the same provider object can
        # be used from successive asyncio.run() calls or worker threads.
        self._loop_clients: weakref.WeakKeyDictionary[asyncio.AbstractEventLoop, httpx.AsyncClient] = (
            weakref.WeakKeyDictionary()
        )
        self._clients_lock = threading.Lock()

    def _auth_headers(self) -> dict[str, str]:
        return {}

    def _client_or_create(self) -> httpx.AsyncClient:
        if self._client is not None:
            return self._client
        loop = asyncio.get_running_loop()
        with self._clients_lock:
            client = self._loop_clients.get(loop)
            if client is None or client.is_closed:
                client = httpx.AsyncClient()
                self._loop_clients[loop] = client
            return client

    async def _request(
        self,
        method: str,
        path: str,
        *,
        payload: Mapping[str, Any] | None = None,
        timeout: float | None = None,
    ) -> httpx.Response:
        url = path if path.startswith(("http://", "https://")) else f"{self.base_url}{path}"
        effective_timeout = self.timeout if timeout is None else timeout
        headers = {**self._auth_headers(), **self._extra_headers}
        try:
            response = await self._client_or_create().request(
                method, url, json=payload, headers=headers, timeout=effective_timeout
            )
        except httpx.TimeoutException as exc:
            raise ProviderUnavailableError(
                f"timed out after {effective_timeout:g}s", provider=self.label
            ) from exc
        except httpx.HTTPError as exc:
            raise ProviderUnavailableError(
                f"transport error: {type(exc).__name__}: {exc}", provider=self.label
            ) from exc
        if response.status_code >= 400:
            raise self._classify_error(response, _error_message(response))
        return response

    def _classify_error(self, response: httpx.Response, message: str) -> ProviderError:
        """Map an error response to an exception. Providers override this for vendor quirks."""
        return error_for_status(
            response.status_code,
            message,
            provider=self.label,
            retry_after=_parse_retry_after(response.headers.get("retry-after")),
        )

    async def _post_json(self, path: str, payload: Mapping[str, Any]) -> dict[str, Any]:
        response = await self._request("POST", path, payload=payload)
        try:
            data = response.json()
        except ValueError as exc:
            raise ResponseParseError("response body is not JSON", provider=self.label) from exc
        if not isinstance(data, dict):
            raise ResponseParseError("response body is not a JSON object", provider=self.label)
        return data

    async def aclose(self) -> None:
        """Close the client owned for the running loop. A client passed in by the caller stays open."""
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return
        with self._clients_lock:
            client = self._loop_clients.pop(loop, None)
        if client is not None:
            await client.aclose()
