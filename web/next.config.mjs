/** @type {import('next').NextConfig} */
const nextConfig = {
  // Static export so the site deploys as plain files (Vercel/any static host).
  // All inference is in-browser; there is no server runtime.
  output: "export",
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
