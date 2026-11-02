"""Model registry and pricing (OpenAI tokens + OpenRouter image rates)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

Provider = Literal["openai", "openrouter"]
PricingMode = Literal["tokens", "per_image", "tiered_image"]


@dataclass(frozen=True)
class ModelConfig:
    id: str
    label: str
    provider: Provider = "openai"
    pricing_mode: PricingMode = "tokens"
    # OpenAI token rates (USD per 1M tokens)
    text_input_per_mtok: float = 0.0
    image_input_per_mtok: float = 0.0
    image_output_per_mtok: float = 0.0
    supports_input_fidelity: bool = False
    # OpenRouter flat / fallback rates
    per_image_usd: float = 0.0
    # resolution -> USD for tiered_image (e.g. Flux / Grok)
    resolution_rates_usd: dict[str, float] = field(default_factory=dict)


MODELS: tuple[ModelConfig, ...] = (
    # --- OpenAI ---
    ModelConfig(
        id="gpt-image-2.5-sunburst",
        label="Flagship (Sunburst)",
        provider="openai",
        pricing_mode="tokens",
        text_input_per_mtok=5.0,
        image_input_per_mtok=8.0,
        image_output_per_mtok=30.0,
    ),
    ModelConfig(
        id="gpt-image-2.5-flare",
        label="Fast / Lightweight (Flare)",
        provider="openai",
        pricing_mode="tokens",
        text_input_per_mtok=5.0,
        image_input_per_mtok=8.0,
        image_output_per_mtok=30.0,
    ),
    ModelConfig(
        id="gpt-image-2",
        label="Standard (GPT Image 2)",
        provider="openai",
        pricing_mode="tokens",
        text_input_per_mtok=5.0,
        image_input_per_mtok=8.0,
        image_output_per_mtok=30.0,
    ),
    ModelConfig(
        id="gpt-image-1.5",
        label="GPT Image 1.5",
        provider="openai",
        pricing_mode="tokens",
        text_input_per_mtok=5.0,
        image_input_per_mtok=8.0,
        image_output_per_mtok=32.0,
        supports_input_fidelity=True,
    ),
    ModelConfig(
        id="gpt-image-1-mini",
        label="GPT Image 1 Mini",
        provider="openai",
        pricing_mode="tokens",
        text_input_per_mtok=2.0,
        image_input_per_mtok=2.5,
        image_output_per_mtok=8.0,
        supports_input_fidelity=False,
    ),
    # --- OpenRouter ---
    ModelConfig(
        id="meta/muse-image",
        label="Meta Muse Image",
        provider="openrouter",
        pricing_mode="per_image",
        per_image_usd=0.01,
    ),
    ModelConfig(
        id="tencent/hy-image-v3.5-preview",
        label="Hy Image 3.5 Preview",
        provider="openrouter",
        pricing_mode="tokens",
        # OpenRouter lists $1.60/M tokens (primarily output)
        image_output_per_mtok=1.60,
    ),
    ModelConfig(
        id="black-forest-labs/flux-3-image",
        label="Flux 3 Image",
        provider="openrouter",
        pricing_mode="tiered_image",
        per_image_usd=0.048,
        resolution_rates_usd={
            "768": 0.041,
            "1K": 0.048,
            "1.5K": 0.07,
            "2K": 0.10,
            "4K": 0.607,
        },
    ),
    ModelConfig(
        id="x-ai/grok-imagine-image-2.0",
        label="Grok Imagine Image 2.0",
        provider="openrouter",
        pricing_mode="tiered_image",
        per_image_usd=0.04,
        resolution_rates_usd={
            "1K": 0.04,
            "1.5K": 0.06,
            "2K": 0.06,
        },
    ),
    ModelConfig(
        id="qwen/qwen-image-2.1-pro",
        label="Qwen Image 2.1 Pro",
        provider="openrouter",
        pricing_mode="tiered_image",
        per_image_usd=0.046,
        resolution_rates_usd={
            "1K": 0.046,
            "2K": 0.046,  # listed from $0.046/image; prefer usage.cost when present
        },
    ),
    ModelConfig(
        id="qwen/qwen-image-2.1-turbo",
        label="Qwen Image 2.1 Turbo",
        provider="openrouter",
        pricing_mode="tiered_image",
        per_image_usd=0.0184,
        resolution_rates_usd={
            "1K": 0.0184,
            "2K": 0.0184,
        },
    ),
)

MODEL_BY_ID = {m.id: m for m in MODELS}


def openrouter_resolution_for_quality(
    quality: str,
    model: ModelConfig | None = None,
) -> str:
    """Map OpenAI-style quality to an OpenRouter resolution the model accepts."""
    q = (quality or "low").strip().lower()
    high = q in {"high", "xhigh", "max"}
    preferred = ("1.5K", "2K", "1K") if high else ("1K", "768", "1.5K", "2K")

    if model and model.resolution_rates_usd:
        for candidate in preferred:
            if candidate in model.resolution_rates_usd:
                return candidate
        return next(iter(model.resolution_rates_usd))

    return "1.5K" if high else "1K"


def calculate_cost_usd(
    model: ModelConfig,
    *,
    text_input_tokens: int = 0,
    image_input_tokens: int = 0,
    image_output_tokens: int = 0,
    total_tokens: int = 0,
    resolution: str | None = None,
    reported_cost_usd: float | None = None,
) -> float:
    """
    Compute USD cost.
    Prefer OpenRouter reported usage.cost when provided.
    """
    if reported_cost_usd is not None:
        return round(float(reported_cost_usd), 6)

    if model.pricing_mode == "per_image":
        return round(model.per_image_usd, 6)

    if model.pricing_mode == "tiered_image":
        res = resolution or "1K"
        rate = model.resolution_rates_usd.get(res, model.per_image_usd)
        return round(rate, 6)

    # tokens mode
    if model.provider == "openrouter" and model.image_output_per_mtok and (
        total_tokens or image_output_tokens
    ):
        toks = total_tokens or (
            text_input_tokens + image_input_tokens + image_output_tokens
        )
        return round((toks / 1_000_000) * model.image_output_per_mtok, 6)

    cost = (
        (text_input_tokens / 1_000_000) * model.text_input_per_mtok
        + (image_input_tokens / 1_000_000) * model.image_input_per_mtok
        + (image_output_tokens / 1_000_000) * model.image_output_per_mtok
    )
    return round(cost, 6)
