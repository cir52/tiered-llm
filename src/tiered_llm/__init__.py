"""tiered-llm: production plumbing for multi-provider LLM systems.

Provider abstraction, per-provider circuit breakers, fallback chains,
cheap-then-expensive cascades and an append-only decision log.
"""

from .breaker import (
    BreakerConfig,
    BreakerRegistry,
    BreakerSnapshot,
    BreakerState,
    CircuitBreaker,
    Permit,
)
from .cascade import Cascade, CascadeItem, CascadeResult, CascadeStats
from .chain import FallbackChain, RetryPolicy
from .decision_log import DecisionLog, iter_decisions
from .errors import (
    AllProvidersFailedError,
    AuthenticationError,
    InvalidRequestError,
    ProviderError,
    ProviderUnavailableError,
    RateLimitError,
    ResponseParseError,
    TieredLLMError,
)
from .health import HealthMonitor
from .jsonparse import JSONExtractionError, parse_json
from .providers import (
    AnthropicProvider,
    DeepSeekProvider,
    GeminiProvider,
    LlamaCppProvider,
    OllamaProvider,
    OpenAICompatibleProvider,
    OpenAIProvider,
    OpenRouterProvider,
    Provider,
    build_provider,
)
from .router import Router
from .types import (
    FREE,
    Attempt,
    CompletionRequest,
    CompletionResponse,
    Message,
    Pricing,
    Usage,
)

__version__ = "0.1.0"

__all__ = [
    "FREE",
    "AllProvidersFailedError",
    "AnthropicProvider",
    "Attempt",
    "AuthenticationError",
    "BreakerConfig",
    "BreakerRegistry",
    "BreakerSnapshot",
    "BreakerState",
    "Cascade",
    "CascadeItem",
    "CascadeResult",
    "CascadeStats",
    "CircuitBreaker",
    "CompletionRequest",
    "CompletionResponse",
    "DecisionLog",
    "DeepSeekProvider",
    "FallbackChain",
    "GeminiProvider",
    "HealthMonitor",
    "InvalidRequestError",
    "JSONExtractionError",
    "LlamaCppProvider",
    "Message",
    "OllamaProvider",
    "OpenAICompatibleProvider",
    "OpenAIProvider",
    "OpenRouterProvider",
    "Permit",
    "Pricing",
    "Provider",
    "ProviderError",
    "ProviderUnavailableError",
    "RateLimitError",
    "ResponseParseError",
    "RetryPolicy",
    "Router",
    "TieredLLMError",
    "Usage",
    "__version__",
    "build_provider",
    "iter_decisions",
    "parse_json",
]
