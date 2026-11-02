"""OpenRouter Images API client: reference-image generation + cost."""

from __future__ import annotations

import base64
import time
from typing import Any

import httpx

from app.models import ModelConfig, calculate_cost_usd, openrouter_resolution_for_quality

OPENROUTER_IMAGES_URL = "https://openrouter.ai/api/v1/images"


def _mime_for_filename(filename: str) -> str:
    ext = filename.rsplit(".", 1)[-1].lower()
    return {
        "png": "image/png",
        "webp": "image/webp",
        "jpg": "image/jpeg",
        "jpeg": "image/jpeg",
    }.get(ext, "application/octet-stream")


def _data_url(filename: str, data: bytes) -> str:
    mime = _mime_for_filename(filename)
    b64 = base64.b64encode(data).decode("ascii")
    return f"data:{mime};base64,{b64}"


def _safe_filename(model_id: str, ext: str = "png") -> str:
    return f"{model_id.replace('/', '_')}.{ext}"


def _extract_usage(usage: Any) -> dict[str, Any]:
    if not isinstance(usage, dict):
        return {
            "text_input_tokens": 0,
            "image_input_tokens": 0,
            "image_output_tokens": 0,
            "total_tokens": 0,
            "reported_cost_usd": None,
        }

    prompt_tokens = int(usage.get("prompt_tokens") or 0)
    completion_tokens = int(usage.get("completion_tokens") or 0)
    total_tokens = int(usage.get("total_tokens") or (prompt_tokens + completion_tokens))
    reported = usage.get("cost")
    reported_cost = float(reported) if reported is not None else None

    return {
        "text_input_tokens": prompt_tokens,
        "image_input_tokens": 0,
        "image_output_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "reported_cost_usd": reported_cost,
        "raw": usage,
    }


async def run_openrouter_image(
    client: httpx.AsyncClient,
    model: ModelConfig,
    *,
    api_key: str,
    person_bytes: bytes,
    person_filename: str,
    apparel_bytes: bytes,
    apparel_filename: str,
    prompt: str,
    quality: str = "low",
    output_format: str = "png",
) -> dict[str, Any]:
    """
    Call OpenRouter POST /api/v1/images with person + apparel references.
    Returns the same result shape as the OpenAI client.
    """
    resolution = openrouter_resolution_for_quality(quality, model)
    aspect_ratio = "2:3"
    started = time.perf_counter()

    base: dict[str, Any] = {
        "model": model.id,
        "label": model.label,
        "provider": model.provider,
        "quality": quality,
        "size": f"{resolution}/{aspect_ratio}",
        "resolution": resolution,
        "aspect_ratio": aspect_ratio,
    }

    payload: dict[str, Any] = {
        "model": model.id,
        "prompt": prompt,
        "n": 1,
        "resolution": resolution,
        "aspect_ratio": aspect_ratio,
        "input_references": [
            {
                "type": "image_url",
                "image_url": {"url": _data_url(person_filename, person_bytes)},
            },
            {
                "type": "image_url",
                "image_url": {"url": _data_url(apparel_filename, apparel_bytes)},
            },
        ],
    }
    # Omit output_format: several OpenRouter providers (e.g. Flux 3) reject it.

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://localhost/virtual-try-on",
        "X-Title": "Virtual Try-On Multi-Model Backend",
    }

    try:
        response = await client.post(
            OPENROUTER_IMAGES_URL,
            headers=headers,
            json=payload,
            timeout=httpx.Timeout(300.0, connect=30.0),
        )
        elapsed_ms = round((time.perf_counter() - started) * 1000, 1)

        if response.status_code >= 400:
            detail = response.text[:800]
            return {
                **base,
                "status": "error",
                "error": f"HTTP {response.status_code}: {detail}",
                "latency_ms": elapsed_ms,
                "usage": None,
                "cost_usd": 0.0,
                "image_bytes": None,
                "output_filename": None,
            }

        body = response.json()
        data = body.get("data") or []
        if not data:
            return {
                **base,
                "status": "error",
                "error": f"Empty data in OpenRouter response: {str(body)[:500]}",
                "latency_ms": elapsed_ms,
                "usage": None,
                "cost_usd": 0.0,
                "image_bytes": None,
                "output_filename": None,
            }

        b64 = data[0].get("b64_json")
        if not b64:
            return {
                **base,
                "status": "error",
                "error": "No b64_json in OpenRouter response",
                "latency_ms": elapsed_ms,
                "usage": None,
                "cost_usd": 0.0,
                "image_bytes": None,
                "output_filename": None,
            }

        image_bytes = base64.b64decode(b64)
        usage = _extract_usage(body.get("usage"))
        cost = calculate_cost_usd(
            model,
            text_input_tokens=usage["text_input_tokens"],
            image_input_tokens=usage["image_input_tokens"],
            image_output_tokens=usage["image_output_tokens"],
            total_tokens=usage["total_tokens"],
            resolution=resolution,
            reported_cost_usd=usage.get("reported_cost_usd"),
        )
        # Drop raw from public usage copy
        public_usage = {
            k: v for k, v in usage.items() if k not in {"raw", "reported_cost_usd"}
        }
        if usage.get("reported_cost_usd") is not None:
            public_usage["reported_cost_usd"] = usage["reported_cost_usd"]

        return {
            **base,
            "status": "ok",
            "error": None,
            "latency_ms": elapsed_ms,
            "usage": public_usage,
            "cost_usd": cost,
            "image_bytes": image_bytes,
            "output_filename": _safe_filename(model.id, output_format if output_format != "jpeg" else "jpg"),
        }
    except Exception as exc:  # noqa: BLE001 — per-model isolation
        elapsed_ms = round((time.perf_counter() - started) * 1000, 1)
        return {
            **base,
            "status": "error",
            "error": f"{type(exc).__name__}: {exc}",
            "latency_ms": elapsed_ms,
            "usage": None,
            "cost_usd": 0.0,
            "image_bytes": None,
            "output_filename": None,
        }
