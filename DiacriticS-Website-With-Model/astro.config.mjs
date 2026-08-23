import { defineConfig } from "astro/config";
import vercel from "@astrojs/vercel";

export default defineConfig({
  site: process.env.PUBLIC_SITE_URL || "https://diacritics.example",
  output: "server",
  adapter: vercel({ maxDuration: 60 }),
  build: {
    inlineStylesheets: "auto",
  },
  vite: {
    server: {
      strictPort: true,
    },
  },
});
