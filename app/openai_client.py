"""OpenAI Images Edits client: run one model and compute cost."""

from __future__ import annotations

import base64
import time
from io import BytesIO
from typing import Any

from openai import AsyncOpenAI

from app.models import ModelConfig, calculate_cost_usd


def _extract_usage(usage: Any) -> dict[str, int]:
    """Normalize Images API usage into text/image input + image output tokens."""
    if usage is None:
        return {
            "text_input_tokens": 0,
            "image_input_tokens": 0,
            "image_output_tokens": 0,
            "total_tokens": 0,
        }

    details = getattr(usage, "input_tokens_details", None)
    text_in = int(getattr(details, "text_tokens", 0) or 0) if details else 0
    image_in = int(getattr(details, "image_tokens", 0) or 0) if details else 0
    output = int(getattr(usage, "output_tokens", 0) or 0)
    total = int(getattr(usage, "total_tokens", 0) or 0)

    # Fallback when details are missing: treat all input as image input.
    if details is None:
        input_tokens = int(getattr(usage, "input_tokens", 0) or 0)
        image_in = input_tokens
        text_in = 0

    return {
        "text_input_tokens": text_in,
        "image_input_tokens": image_in,
        "image_output_tokens": output,
        "total_tokens": total,
    }


def _safe_filename(model_id: str, ext: str = "png") -> str:
    return f"{model_id.replace('/', '_')}.{ext}"


async def run_image_edit(
    client: AsyncOpenAI,
    model: ModelConfig,
    *,
    person_bytes: bytes,
    person_filename: str,
    apparel_bytes: bytes,
    apparel_filename: str,
    prompt: str,
    quality: str = "low",
    size: str = "1024x1536",
    output_format: str = "png",
) -> dict[str, Any]:
    """
    Call /v1/images/edits for one model.
    Returns a result dict with status, cost, latency, and image bytes on success.
    """
    started = time.perf_counter()

    def _file_tuple(filename: str, data: bytes) -> tuple[str, BytesIO, str]:
        ext = filename.rsplit(".", 1)[-1].lower()
        mime = {
            "png": "image/png",
            "webp": "image/webp",
            "jpg": "image/jpeg",
            "jpeg": "image/jpeg",
        }.get(ext, "application/octet-stream")
        return (filename, BytesIO(data), mime)

    kwargs: dict[str, Any] = {
        "model": model.id,
        "label": model.label,
        "provider": model.provider,
        "quality": quality,
        "size": size,
    }

    try:
        edit_kwargs: dict[str, Any] = {
            "model": model.id,
            "image": [
                _file_tuple(person_filename, person_bytes),
                _file_tuple(apparel_filename, apparel_bytes),
            ],
            "prompt": prompt,
            "quality": quality,
            "size": size,
            "n": 1,
            "output_format": output_format,
        }
        if model.supports_input_fidelity:
            edit_kwargs["input_fidelity"] = "high"

        response = await client.images.edit(**edit_kwargs)
        elapsed_ms = round((time.perf_counter() - started) * 1000, 1)

        if not response.data:
            return {
                **kwargs,
                "status": "error",
                "error": "Empty response.data from Images API",
                "latency_ms": elapsed_ms,
                "usage": None,
                "cost_usd": 0.0,
                "image_bytes": None,
                "output_filename": None,
            }

        b64 = response.data[0].b64_json
        if not b64:
            return {
                **kwargs,
                "status": "error",
                "error": "No b64_json in response (URL-only responses are unsupported)",
                "latency_ms": elapsed_ms,
                "usage": None,
                "cost_usd": 0.0,
                "image_bytes": None,
                "output_filename": None,
            }

        image_bytes = base64.b64decode(b64)
        usage = _extract_usage(getattr(response, "usage", None))
        cost = calculate_cost_usd(
            model,
            text_input_tokens=usage["text_input_tokens"],
            image_input_tokens=usage["image_input_tokens"],
            image_output_tokens=usage["image_output_tokens"],
        )
        out_name = _safe_filename(model.id, output_format)

        return {
            **kwargs,
            "status": "ok",
            "error": None,
            "latency_ms": elapsed_ms,
            "usage": usage,
            "cost_usd": cost,
            "image_bytes": image_bytes,
            "output_filename": out_name,
        }
    except Exception as exc:  # noqa: BLE001 — per-model isolation
        elapsed_ms = round((time.perf_counter() - started) * 1000, 1)
        return {
            **kwargs,
            "status": "error",
            "error": f"{type(exc).__name__}: {exc}",
            "latency_ms": elapsed_ms,
            "usage": None,
            "cost_usd": 0.0,
            "image_bytes": None,
            "output_filename": None,
        }
