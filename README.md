# Virtual Try-On

**Multi-model AI virtual try-on benchmark:** run the same person, apparel, and prompt across OpenAI + OpenRouter image models, then compare visual quality, latency, and per-model cost side by side.

## Why this exists

Choosing an image model for fashion try-on is not just a quality call — it is a product tradeoff across fidelity, identity preservation, speed, and unit cost. Vendors ship different APIs, pricing shapes, and quality knobs, which makes apples-to-apples comparison slow.

This project turns that into a repeatable experiment: one person image, one apparel image, one shared markdown prompt, many models — with every PNG and dollar cost written to a single local run folder.

## Who it is for

- AI / ML product managers evaluating try-on model candidates
- Engineering teams validating OpenAI vs OpenRouter image stacks before production
- Anyone who needs a cost-and-quality matrix, not a single demo screenshot

## What you get

| Capability | Detail |
| --- | --- |
| Same inputs everywhere | Person + apparel + identical prompt |
| Dual providers | OpenAI Images Edits + OpenRouter Images API |
| Parallel runs | All registered models in one `POST /try-on` |
| Cost visibility | Per-model `cost_usd` + `total_cost_usd` in `results.json` |
| Local review | Open `outputs/<run>/` and click PNGs — no UI required |

## Model matrix

### OpenAI (Images Edits)

| Model | Role |
| --- | --- |
| `gpt-image-2.5-sunburst` | Flagship / high-fidelity |
| `gpt-image-2.5-flare` | Fast / lightweight |
| `gpt-image-2` | Standard |
| `gpt-image-1.5` | GPT Image 1.5 |
| `gpt-image-1-mini` | Cheaper / lighter |

### OpenRouter (Images API)

| Model | Role | Pricing (approx.) |
| --- | --- | --- |
| `meta/muse-image` | Meta Muse Image | $0.01 / image |
| `tencent/hy-image-v3.5-preview` | Hy Image 3.5 Preview | $1.60 / M tokens |
| `black-forest-labs/flux-3-image` | Flux 3 Image | ~$0.048 / image at 1K |
| `x-ai/grok-imagine-image-2.0` | Grok Imagine Image 2.0 | ~$0.04 / image at 1K |
| `qwen/qwen-image-2.1-pro` | Qwen Image 2.1 Pro | ~$0.046 / image at 1K |
| `qwen/qwen-image-2.1-turbo` | Qwen Image 2.1 Turbo | ~$0.0184 / image at 1K |

**Notes**

- Default quality is `low` (OpenAI). OpenRouter maps to `aspect_ratio=2:3` and `resolution=1K` (higher quality maps to `1.5K` / `2K` when the model supports it).
- OpenRouter cost prefers API `usage.cost` when present.
- `meta/muse-image` may return HTTP 403 in some regions (geo restriction on OpenRouter).

## Quick start

```bash
cd Virtual-Try-On
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# Edit .env locally — set OPENAI_API_KEY and OPENROUTER_API_KEY
# Never commit .env
```

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

```bash
curl -X POST "http://localhost:8000/try-on" \
  -F "person=@./samples/person.jpg" \
  -F "apparel=@./samples/apparel.jpg" \
  -F "quality=low"
```

Optional form fields: `quality` (default `low`), `size` (OpenAI only, default `1024x1536`).

## Outputs

Each run creates `outputs/<UTC-timestamp>/` (gitignored):

| File | Description |
| --- | --- |
| `person.*` / `apparel.*` | Copies of inputs |
| `prompt.md` | Exact prompt sent to every model |
| `<model>.png` | One image per successful model |
| `results.json` | Status, latency, usage, `cost_usd`, `total_cost_usd` |

Prompts live in `prompts/virtual_try_on.md` and `prompts/negative.md` — edit them to change the experiment without touching code.

## API

- `GET /health` — liveness
- `GET /models` — registered models, providers, pricing metadata
- `POST /try-on` — multipart `person` + `apparel` (+ optional `quality`, `size`)

## Security

- API keys stay in local `.env` only.
- `.env`, virtualenvs, run outputs, and personal sample photos are gitignored.
- Commit `.env.example` placeholders — never real keys.

## License

Use and extend for evaluation and product research as needed for your organization.
