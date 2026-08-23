import type { APIRoute } from "astro";

export const prerender = false;

type RateRecord = { count: number; resetAt: number };
const rateRecords = new Map<string, RateRecord>();
const MAX_BODY_BYTES = 32_768;
const MAX_TEXT_CHARACTERS = 5_000;
const astroEnv = import.meta.env as Record<string, string | undefined>;
const env = (name: string, fallback = "") => astroEnv[name] ?? process.env[name] ?? fallback;

const curated = new Map([
  ["اللغة العربية جميلة", "اَللُّغَةُ الْعَرَبِيَّةُ جَمِيلَةٌ"],
  ["يكتب الباحث نتائج التجربة", "يَكْتُبُ الْبَاحِثُ نَتَائِجَ التَّجْرِبَةِ"],
  ["هذا نموذج متخصص في اللغة العربية", "هَذَا نَمُوذَجٌ مُتَخَصِّصٌ فِي اللُّغَةِ الْعَرَبِيَّةِ"],
]);

const marks = /[\u0610-\u061A\u064B-\u065F\u0670\u06D6-\u06ED]/g;
const normalizeSkeleton = (text: string) => text.normalize("NFC").replace(marks, "").replace(/\s+/g, " ").trim();
const json = (body: unknown, status = 200) =>
  new Response(JSON.stringify(body), {
    status,
    headers: {
      "content-type": "application/json; charset=utf-8",
      "cache-control": "no-store",
      "x-content-type-options": "nosniff",
    },
  });

function isRateLimited(ip: string) {
  const now = Date.now();
  const limit = Number.parseInt(env("DIACRITICS_RATE_LIMIT", "10"), 10);
  const existing = rateRecords.get(ip);
  if (!existing || existing.resetAt <= now) {
    rateRecords.set(ip, { count: 1, resetAt: now + 60_000 });
    return false;
  }
  existing.count += 1;
  return existing.count > limit;
}

export const POST: APIRoute = async ({ request }) => {
  const requestId = crypto.randomUUID();
  const ip = request.headers.get("x-forwarded-for")?.split(",")[0]?.trim() || "local";
  if (isRateLimited(ip)) return json({ code: "RATE_LIMITED", requestId }, 429);
  const contentLength = Number.parseInt(request.headers.get("content-length") || "0", 10);
  if (contentLength > MAX_BODY_BYTES) return json({ code: "TEXT_TOO_LONG", requestId }, 413);

  let payload: { text?: unknown; locale?: unknown };
  try {
    payload = await request.json();
  } catch {
    return json({ code: "UNSUPPORTED_TEXT", requestId }, 400);
  }

  if (typeof payload.text !== "string" || !payload.text.trim()) return json({ code: "EMPTY_TEXT", requestId }, 400);
  const text = payload.text.normalize("NFC").trim();
  // Cheap abuse guard only. The Modal endpoint applies the authoritative Qwen tokenizer limit:
  // 383 input tokens / 768 tokens after the tagger's internal double pass.
  if (Array.from(text).length > MAX_TEXT_CHARACTERS) return json({ code: "TEXT_TOO_LONG", requestId }, 413);
  if (!/[\u0621-\u064A]/.test(text)) return json({ code: "UNSUPPORTED_TEXT", requestId }, 400);

  const endpoint = env("DIACRITICS_GPU_ENDPOINT");
  const token = env("DIACRITICS_GPU_TOKEN");
  const modelLabel = env("DIACRITICS_MODEL_LABEL", "Character tagger · local configuration");
  const mockMode = env("DIACRITICS_MOCK_MODE", "curated");
  const started = performance.now();

  if (!endpoint) {
    if (mockMode === "timeout") return json({ code: "UPSTREAM_TIMEOUT", requestId }, 504);
    if (mockMode === "error") return json({ code: "UPSTREAM_ERROR", requestId }, 502);
    const output = mockMode === "curated" ? curated.get(normalizeSkeleton(text)) : undefined;
    if (!output) return json({ code: "MODEL_NOT_CONFIGURED", requestId }, 503);
    return json({
      input: text,
      output,
      modelLabel: "Curated local interface preview",
      durationMs: performance.now() - started,
      requestId,
    });
  }

  const controller = new AbortController();
  const timeoutMs = Number.parseInt(env("DIACRITICS_TIMEOUT_MS", "60000"), 10);
  const timeout = setTimeout(() => controller.abort(), timeoutMs);
  try {
    const normalizedEndpoint = endpoint.replace(/\/$/, "");
    const inferenceUrl = normalizedEndpoint.endsWith("/v1/diacritize")
      ? normalizedEndpoint
      : `${normalizedEndpoint}/v1/diacritize`;
    const upstream = await fetch(inferenceUrl, {
      method: "POST",
      headers: {
        "content-type": "application/json",
        ...(token ? { authorization: `Bearer ${token}` } : {}),
      },
      body: JSON.stringify({ text }),
      signal: controller.signal,
    });
    const data = await upstream.json().catch(() => ({})) as {
      result?: unknown;
      output?: unknown;
      diacritized?: unknown;
      detail?: unknown;
    };
    if (!upstream.ok) {
      if (upstream.status === 413) return json({ code: "TEXT_TOO_LONG", requestId, limits: data.detail }, 413);
      if (upstream.status === 401 || upstream.status === 403) return json({ code: "MODEL_AUTH_ERROR", requestId }, 502);
      return json({ code: "UPSTREAM_ERROR", requestId }, 502);
    }
    const output = typeof data.result === "string"
      ? data.result
      : typeof data.output === "string"
        ? data.output
        : data.diacritized;
    if (typeof output !== "string" || !output.trim()) return json({ code: "UPSTREAM_ERROR", requestId }, 502);
    return json({ input: text, output, modelLabel, durationMs: performance.now() - started, requestId });
  } catch (error) {
    if (error instanceof DOMException && error.name === "AbortError") return json({ code: "UPSTREAM_TIMEOUT", requestId }, 504);
    return json({ code: "UPSTREAM_ERROR", requestId }, 502);
  } finally {
    clearTimeout(timeout);
  }
};
