import { execSync } from "node:child_process";

/**
 * Resolve the commit SHA at build time so the footer can show what's live.
 * Order: Vercel's auto-injected var → a CI-provided var → local git → "dev".
 * Evaluated once at config load; the value is baked into the static export.
 */
function resolveBuildSha() {
  const fromEnv =
    process.env.VERCEL_GIT_COMMIT_SHA || process.env.NEXT_PUBLIC_BUILD_SHA;
  if (fromEnv) return fromEnv;
  try {
    return execSync("git rev-parse HEAD", { stdio: ["ignore", "pipe", "ignore"] })
      .toString()
      .trim();
  } catch {
    return "dev";
  }
}

const BUILD_SHA = resolveBuildSha();
const BUILD_DATE = new Date().toISOString();

/** @type {import('next').NextConfig} */
const nextConfig = {
  // Static export so the site deploys as plain files (Vercel/any static host).
  // All inference is in-browser; there is no server runtime.
  output: "export",
  // Baked-in build provenance (see lib/build-info.ts + components/footer.tsx).
  env: {
    NEXT_PUBLIC_BUILD_SHA: BUILD_SHA,
    NEXT_PUBLIC_BUILD_DATE: BUILD_DATE,
  },
  images: { unoptimized: true },
  reactStrictMode: true,
  // onnxruntime-web ships .wasm/.mjs that must not be bundled/parsed by webpack.
  // Mark it external on the server build and let the browser load it directly.
  webpack: (config, { isServer }) => {
    if (!isServer) {
      config.resolve.fallback = { ...config.resolve.fallback, fs: false, path: false };
    }
    return config;
  },
};

export default nextConfig;
