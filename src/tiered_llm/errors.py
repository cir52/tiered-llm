"""Exception hierarchy.

Providers never return error dictionaries: every failure is a typed exception,
and the type decides what the fallback chain does with it.

* ``retryable = True``  -> the breaker counts it and the chain moves on to the
  next provider (rate limits, outages, timeouts, bad credentials, garbage output).
* ``retryable = False`` -> the request itself is the problem (malformed payload,
  context too long). Another provider would most likely reject it too, so the
  chain re-raises immediately and the breaker is left untouched.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .types import Attempt


class TieredLLMError(Exception):
    """Base class for every error raised by this package."""


class ProviderError(TieredLLMError):
    """A single provider call failed."""

    retryable: bool = True

    def __init__(
        self,
        message: str,
        *,
        provider: str = "",
        status_code: int | None = None,
        retry_after: float | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.provider = provider
        self.status_code = status_code
        self.retry_after = retry_after

    def __str__(self) -> str:
        prefix = f"[{self.provider}] " if self.provider else ""
        status = f" (HTTP {self.status_code})" if self.status_code else ""
        return f"{prefix}{self.message}{status}"


class RateLimitError(ProviderError):
    """HTTP 429. If the provider sent ``Retry-After`` the breaker opens for that long."""


class ProviderUnavailableError(ProviderError):
    """Timeouts, connection errors, 5xx, overloaded, out of credits, unknown model."""


class AuthenticationError(ProviderError):
    """Rejected credentials. Retryable because the *next* provider has its own key."""


class ResponseParseError(ProviderError):
    """The provider answered, but not with something we can use."""


class InvalidRequestError(ProviderError):
    """The request is malformed or too large. Not retried, does not trip breakers."""

    retryable = False


class AllProvidersFailedError(TieredLLMError):
    """Every provider in a chain failed or was skipped because its circuit was open."""

    def __init__(self, chain: str, attempts: Sequence[Attempt]) -> None:
        self.chain = chain
        self.attempts = tuple(attempts)
        detail = "; ".join(f"{a.provider}: {a.outcome} ({a.error})" for a in self.attempts)
        super().__init__(f"all providers failed in chain '{chain}': {detail or 'no providers'}")


def error_for_status(
    status: int,
    message: str,
    *,
    provider: str,
    retry_after: float | None = None,
) -> ProviderError:
    """Map an HTTP status code to the matching exception type."""
    if status == 429:
        return RateLimitError(message, provider=provider, status_code=status, retry_after=retry_after)
    if status in (401, 403):
        return AuthenticationError(message, provider=provider, status_code=status)
    if status in (400, 413, 422):
        return InvalidRequestError(message, provider=provider, status_code=status)
    if status == 404:
        return ProviderUnavailableError(
            f"not found - wrong model name or base_url? {message}",
            provider=provider,
            status_code=status,
        )
    # 402 (no credits), 408, 409, 5xx, 529 (overloaded) ...
    return ProviderUnavailableError(message, provider=provider, status_code=status, retry_after=retry_after)
