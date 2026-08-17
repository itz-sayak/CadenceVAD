import cloudflare from "@astrojs/cloudflare";
import react from "@astrojs/react";
import sitemap from "@astrojs/sitemap";
import { defineConfig } from "astro/config";

export default defineConfig({
  adapter: cloudflare({ imageService: "compile" }),
  base: process.env.CADENCEVAD_BASE ?? "/",
  integrations: [
    react(),
    sitemap({
      namespaces: {
        image: false,
        news: false,
        video: false,
        xhtml: false,
      },
      serialize(item) {
        item.changefreq = "weekly";
        item.priority = 1;
        return item;
      },
    }),
  ],
  output: "server",
  site: "https://itz-sayak.github.io",
  trailingSlash: "always",
  vite: {
    optimizeDeps: {
      include: ["@astrojs/internal-helpers > picomatch"],
    },
  },
});
