import type { NextConfig } from "next";

// Static export: the whole app is plain files on a free static host (Cloudflare Pages).
const nextConfig: NextConfig = {
  output: "export",
  trailingSlash: true,
  images: { unoptimized: true },
  // styled, bilingual 404 for URLs that match no route (src/app/global-not-found.tsx)
  experimental: { globalNotFound: true },
  turbopack: {
    rules: {
      "*.css": {
        loaders: ["@tailwindcss/turbopack"],
        as: "*.css",
      },
    },
  },
};

export default nextConfig;
