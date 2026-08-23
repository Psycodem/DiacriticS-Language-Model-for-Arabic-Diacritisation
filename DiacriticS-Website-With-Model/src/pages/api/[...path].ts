import type { APIRoute } from "astro";

export const prerender = false;
export const ALL: APIRoute = () =>
  new Response(JSON.stringify({ code: "NOT_FOUND" }), {
    status: 404,
    headers: { "content-type": "application/json; charset=utf-8", "cache-control": "no-store" },
  });
