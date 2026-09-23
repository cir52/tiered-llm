"""Provider implementations and the config-driven factory."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from ..breaker import BreakerConfig
from ..types import Pricing
from .anthropic import AnthropicProvider
from .base import HTTPProvider, Provider
from .gemini import GeminiProvider
from .ollama import OllamaProvider, choose_model
from .openai_compat import (
    DeepSeekProvider,
    LlamaCppProvider,
    OpenAICompatibleProvider,
    OpenAIProvider,
    OpenRouterProvider,
)

__all__ = [
    "PROVIDER_TYPES",
    "AnthropicProvider",
    "DeepSeekProvider",
    "GeminiProvider",
    "HTTPProvider",
    "LlamaCppProvider",
    "OllamaProvider",
    "OpenAICompatibleProvider",
    "OpenAIProvider",
    "OpenRouterProvider",
    "Provider",
    "build_provider",
    "choose_model",
]

PROVIDER_TYPES: dict[str, type[Provider]] = {
    cls.kind: cls
    for cls in (
        AnthropicProvider,
        GeminiProvider,
        OpenAIProvider,
        DeepSeekProvider,
        OpenRouterProvider,
        OpenAICompatibleProvider,
        LlamaCppProvider,
        OllamaProvider,
    )
}


def build_provider(label: str, spec: Mapping[str, Any]) -> Provider:
    """Create a provider from a plain config mapping.

    ``{"type": "anthropic", "model": "...", "pricing": {...}, "breaker": {...}}``

    Secrets are deliberately not accepted here: reference an environment
    variable with ``api_key_env`` so that config files can be committed.
    """
    options = dict(spec)
    if "api_key" in options:
        raise ValueError(f"provider '{label}': do not put API keys in config - use api_key_env instead")
    kind = options.pop("type", None)
    if kind not in PROVIDER_TYPES:
        known = ", ".join(sorted(PROVIDER_TYPES))
        raise ValueError(f"provider '{label}': unknown type {kind!r} (known: {known})")
    model = options.pop("model", None)
    if not model and kind != "llamacpp":
        raise ValueError(f"provider '{label}': 'model' is required")
    if "pricing" in options:
        options["pricing"] = Pricing(**options["pricing"])
    if "breaker" in options:
        options["breaker_config"] = BreakerConfig(**options.pop("breaker"))
    options.setdefault("label", label)
    cls: Any = PROVIDER_TYPES[kind]
    if model:
        return cls(model, **options)  # type: ignore[no-any-return]
    return cls(**options)  # type: ignore[no-any-return]
