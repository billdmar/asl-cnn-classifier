import type { MetadataRoute } from "next";

/**
 * Pure PWA manifest object for the ASL Classifier.
 *
 * Exported separately from `app/manifest.ts` so it can be imported and asserted
 * against in unit tests without pulling in Next's route machinery.
 *
 * Brand: dark "AI product" — background/theme #0a0a0f. The app has a light/dark
 * toggle but defaults to dark, so the manifest advertises the dark chrome.
 */
export const manifest: MetadataRoute.Manifest = {
  name: "ASL Classifier",
  short_name: "ASL Classifier",
  description:
    "Real-time American Sign Language alphabet recognition running 100% in your browser — webcam frames never leave your device.",
  start_url: "/",
  scope: "/",
  display: "standalone",
  background_color: "#0a0a0f",
  theme_color: "#0a0a0f",
  categories: ["education", "utilities", "productivity"],
  icons: [
    {
      src: "/icons/icon-192.png",
      sizes: "192x192",
      type: "image/png",
      purpose: "any",
    },
    {
      src: "/icons/icon-512.png",
      sizes: "512x512",
      type: "image/png",
      purpose: "any",
    },
    {
      src: "/icons/icon-512-maskable.png",
      sizes: "512x512",
      type: "image/png",
      purpose: "maskable",
    },
  ],
};
