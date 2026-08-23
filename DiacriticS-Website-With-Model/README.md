# DiacriticS Research Experience

Clean-room bilingual research website and model interface for DiacriticS. This directory was created from the approved `DIACRITICS_BUILD_BRIEF.md`; it does not import or depend on any existing website implementation in the repository.

## Local setup

Requirements: Node.js 22+ and npm.

```bash
npm install
cp .env.example .env
npm run dev
```

Open `http://localhost:4321`. Routes:

- `/` — English research report
- `/ar/` — independently composed Arabic RTL report
- `/demo/` — English demo interface
- `/ar/demo/` — Arabic demo interface

Run a production check and local preview:

```bash
npm run build
npm run preview
```

## Demo boundary

The browser posts only `{ text, locale }` to `/api/diacritize`. The server validates the request, applies a cheap request-size guard and a per-minute rate limit, then either:

1. forwards the request to `DIACRITICS_GPU_ENDPOINT/v1/diacritize` with the server-only bearer token; or
2. uses the curated local interface mock when no endpoint is configured.

`DIACRITICS_GPU_ENDPOINT` and `DIACRITICS_GPU_TOKEN` must never use the `PUBLIC_` prefix. The browser bundle does not contain either value. The curated mock returns results only for the three interface examples; it never fabricates arbitrary model output.

The production model applies the authoritative tokenizer-aware context limit: 383 input tokens
(typically about 1,130 Arabic characters) and 768 tokens after its internal double pass. A cold
request after the Modal container has scaled to zero may take about 44 seconds, so the server
timeout is 60 seconds.

Use `DIACRITICS_MOCK_MODE=timeout` or `error` to verify failure states. Use `disabled` to verify the unconfigured-model response.

## Vercel deployment

Use `DiacriticS-Research-Experience` as the Vercel project root. The Astro Vercel adapter emits
the server function with a 60-second maximum duration, which covers the measured Modal cold
start while keeping the bearer credential on the server.

Set these Production environment variables in Vercel before redeploying:

```text
PUBLIC_SITE_URL=https://diacritics.vercel.app
DIACRITICS_GPU_ENDPOINT=https://<your-modal-app>.modal.run
DIACRITICS_GPU_TOKEN=<dedicated DiacriticS API token>
DIACRITICS_MODEL_LABEL=Qwen3.5-0.8B char tagger · full epoch
DIACRITICS_TIMEOUT_MS=60000
DIACRITICS_RATE_LIMIT=10
DIACRITICS_MOCK_MODE=disabled
```

Do not configure an uptime monitor against either the website API or Modal health endpoint: it
would keep waking the scale-to-zero GPU.

## Research content

Research content is centralized in `src/content/site.ts`. Values carry visible provisional/scorer labels in the interface. Before publication, replace the pending Gemma/Qwen and zero-shot content only with an authorized data record that identifies track, checkpoint, dataset/split, scorer generation, metric variant, CA/MSA values, macro average, date, and source.

Do not compare the generation-one baseline table directly with corrected fine-tuning runs. Completed runs have not performed decontamination. Public values are rounded to one decimal; authoritative precision belongs in the source record.

## Publication blockers

- Team-approved English and human-authored Arabic analysis copy
- Final authorized results and provenance
- Team credits, contact destination, and repository URL
- Usage permission and credit wording for both displayed architectural images
- Confirmed web-use licence for the locally hosted Thmanyah Sans files (sourced from the pinned community package repository)
- Final GPU endpoint contract, public model label, and operational limits
- Final canonical domain, after local approval

The current `robots.txt`, sitemap, and canonical origin intentionally use `https://diacritics.example`; replace them for preview/production. SVG Open Graph cards are included for local review and should be exported to 1200×630 PNGs before public launch for broad crawler compatibility.
