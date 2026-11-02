"""Virtual Try-On multi-model backend — FastAPI entrypoint."""

from __future__ import annotations

import asyncio
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from openai import AsyncOpenAI

from app.models import MODELS
from app.openai_client import run_image_edit
from app.openrouter_client import run_openrouter_image
from app.prompt_loader import load_try_on_prompt

load_dotenv()

ROOT = Path(__file__).resolve().parent.parent
OUTPUTS_DIR = ROOT / "outputs"

app = FastAPI(
    title="Virtual Try-On Multi-Model Backend",
    description=(
        "Upload a person image and apparel image; runs the same markdown prompt "
        "across OpenAI and OpenRouter image models and writes results + costs to outputs/."
    ),
    version="1.1.0",
)


def _ext_from_upload(upload: UploadFile, fallback: str = "jpg") -> str:
    name = upload.filename or ""
    suffix = Path(name).suffix.lstrip(".").lower()
    if suffix in {"jpg", "jpeg", "png", "webp"}:
        return "jpg" if suffix == "jpeg" else suffix
    content_type = (upload.content_type or "").lower()
    if "png" in content_type:
        return "png"
    if "webp" in content_type:
        return "webp"
    if "jpeg" in content_type or "jpg" in content_type:
        return "jpg"
    return fallback


def _public_result(result: dict[str, Any]) -> dict[str, Any]:
    """Strip binary payload before JSON serialization."""
    out: dict[str, Any] = {
        "model": result["model"],
        "label": result["label"],
        "provider": result.get("provider"),
        "status": result["status"],
        "error": result["error"],
        "latency_ms": result["latency_ms"],
        "quality": result["quality"],
        "size": result["size"],
        "usage": result["usage"],
        "cost_usd": result["cost_usd"],
        "output_filename": result["output_filename"],
    }
    if "resolution" in result:
        out["resolution"] = result["resolution"]
    if "aspect_ratio" in result:
        out["aspect_ratio"] = result["aspect_ratio"]
    return out


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/models")
async def list_models() -> dict[str, Any]:
    return {
        "models": [
            {
                "id": m.id,
                "label": m.label,
                "provider": m.provider,
                "pricing_mode": m.pricing_mode,
                "pricing": {
                    "text_input_per_mtok": m.text_input_per_mtok,
                    "image_input_per_mtok": m.image_input_per_mtok,
                    "image_output_per_mtok": m.image_output_per_mtok,
                    "per_image_usd": m.per_image_usd,
                    "resolution_rates_usd": m.resolution_rates_usd,
                },
                "supports_input_fidelity": m.supports_input_fidelity,
            }
            for m in MODELS
        ]
    }


@app.post("/try-on")
async def try_on(
    person: UploadFile = File(..., description="Person / headshot reference"),
    apparel: UploadFile = File(..., description="Apparel / clothing reference"),
    quality: str = Form("low"),
    size: str = Form("1024x1536"),
) -> dict[str, Any]:
    openai_key = os.getenv("OPENAI_API_KEY")
    openrouter_key = os.getenv("OPENROUTER_API_KEY")
    if not openai_key:
        raise HTTPException(
            status_code=500,
            detail="OPENAI_API_KEY is not set. Copy .env.example to .env and add your key.",
        )
    if not openrouter_key:
        raise HTTPException(
            status_code=500,
            detail="OPENROUTER_API_KEY is not set. Copy .env.example to .env and add your key.",
        )

    person_bytes = await person.read()
    apparel_bytes = await apparel.read()
    if not person_bytes:
        raise HTTPException(status_code=400, detail="person file is empty")
    if not apparel_bytes:
        raise HTTPException(status_code=400, detail="apparel file is empty")

    try:
        prompt = load_try_on_prompt()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=500, detail=f"Prompt file missing: {exc}") from exc

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = OUTPUTS_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    person_ext = _ext_from_upload(person)
    apparel_ext = _ext_from_upload(apparel)
    person_name = f"person.{person_ext}"
    apparel_name = f"apparel.{apparel_ext}"

    (run_dir / person_name).write_bytes(person_bytes)
    (run_dir / apparel_name).write_bytes(apparel_bytes)
    (run_dir / "prompt.md").write_text(prompt, encoding="utf-8")

    openai_client = AsyncOpenAI(api_key=openai_key)
    tasks: list[Any] = []

    async with httpx.AsyncClient() as http_client:
        for model in MODELS:
            if model.provider == "openai":
                tasks.append(
                    run_image_edit(
                        openai_client,
                        model,
                        person_bytes=person_bytes,
                        person_filename=person_name,
                        apparel_bytes=apparel_bytes,
                        apparel_filename=apparel_name,
                        prompt=prompt,
                        quality=quality,
                        size=size,
                        output_format="png",
                    )
                )
            else:
                tasks.append(
                    run_openrouter_image(
                        http_client,
                        model,
                        api_key=openrouter_key,
                        person_bytes=person_bytes,
                        person_filename=person_name,
                        apparel_bytes=apparel_bytes,
                        apparel_filename=apparel_name,
                        prompt=prompt,
                        quality=quality,
                        output_format="png",
                    )
                )

        raw_results = await asyncio.gather(*tasks)

    model_results: list[dict[str, Any]] = []
    total_cost = 0.0
    for result in raw_results:
        if result["status"] == "ok" and result["image_bytes"] and result["output_filename"]:
            (run_dir / result["output_filename"]).write_bytes(result["image_bytes"])
            total_cost += float(result["cost_usd"] or 0.0)
        model_results.append(_public_result(result))

    summary = {
        "run_id": run_id,
        "output_dir": str(run_dir),
        "quality": quality,
        "size": size,
        "prompt_files": [
            "prompts/virtual_try_on.md",
            "prompts/negative.md",
        ],
        "inputs": {
            "person": person_name,
            "apparel": apparel_name,
        },
        "models": model_results,
        "total_cost_usd": round(total_cost, 6),
    }

    (run_dir / "results.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    return summary
