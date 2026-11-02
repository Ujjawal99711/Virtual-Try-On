"""Load virtual try-on prompts from markdown files."""

from __future__ import annotations

from pathlib import Path

PROMPTS_DIR = Path(__file__).resolve().parent.parent / "prompts"


def load_try_on_prompt(
    prompts_dir: Path | None = None,
) -> str:
    """Concatenate positive + negative markdown into one prompt string."""
    base = prompts_dir or PROMPTS_DIR
    positive = (base / "virtual_try_on.md").read_text(encoding="utf-8").strip()
    negative = (base / "negative.md").read_text(encoding="utf-8").strip()
    return f"{positive}\n\n---\n\n{negative}\n"
