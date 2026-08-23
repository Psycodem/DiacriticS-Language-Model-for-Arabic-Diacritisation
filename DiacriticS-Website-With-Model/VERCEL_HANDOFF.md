# DiacriticS Vercel deployment handoff

Please replace the current Vercel deployment with the newer Andalusian-themed Astro website.

## Project configuration

Use this repository directory as the Vercel project root:

```text
DiacriticS-Website-With-Model
```

Use the following build settings:

```text
Framework: Astro
Install command: npm install
Build command: npm run build
Output directory: detected automatically by @astrojs/vercel
Node.js version: 24
```

The repository already uses `@astrojs/vercel` and configures the serverless function with a
60-second maximum duration. Do not remove that setting.

## Production environment variables

Add these variables to the Vercel Production environment before deploying:

```text
PUBLIC_SITE_URL=https://diacritics.vercel.app
DIACRITICS_GPU_ENDPOINT=https://<your-modal-app>.modal.run
DIACRITICS_GPU_TOKEN=<SEND SEPARATELY THROUGH A PRIVATE CHANNEL>
DIACRITICS_MODEL_LABEL=Qwen3.5-0.8B char tagger · full epoch
DIACRITICS_TIMEOUT_MS=60000
DIACRITICS_RATE_LIMIT=10
DIACRITICS_MOCK_MODE=disabled
```

`DIACRITICS_GPU_TOKEN` is a secret. It must remain server-side:

- Never prefix it with `PUBLIC_`.
- Never put it in client-side JavaScript, HTML, or a public repository file.
- Do not send it in a shared issue or alongside this document.
- The browser must call the same-origin `/api/diacritize` route, not Modal directly.

## Routes to verify

After deployment, verify these routes:

```text
/             English research website
/ar/          Arabic RTL research website
/demo/        English live model interface
/ar/demo/     Arabic live model interface
/api/diacritize  Server-side inference proxy
```

## Functional verification

Submit this text through `/demo/`:

```text
العلم نور والجهل ظلام
```

Expected output:

```text
الْعِلْمُ نُورٌ وَالْجَهْلُ ظَلَامٌ
```

Also verify:

1. The browser Network panel calls only `/api/diacritize`.
2. The Modal endpoint and bearer token do not appear in downloaded HTML or JavaScript.
3. English and Arabic pages work at a 375-pixel mobile width.
4. Inputs beyond the model's 383-token context return a readable limit message.
5. A request after more than two minutes of inactivity succeeds. Modal may need approximately
   44 seconds to start the GPU and load the model.
6. A second warm request completes much faster.

## Cost protection

- Do not configure uptime monitoring or periodic health checks for the website API or Modal URL.
  Those checks would repeatedly start the GPU.
- The Modal deployment is configured with zero minimum containers, one maximum container, and a
  120-second idle shutdown window.
- Do not expose the Modal endpoint credential publicly.

## Deployment completion

Keep the existing public origin:

```text
https://diacritics.vercel.app
```

After deployment, send back:

- the production deployment URL;
- confirmation that all four pages load;
- the result of the Arabic test above;
- confirmation that the token is absent from browser assets;
- a screenshot of the English and Arabic demo pages at mobile width.

