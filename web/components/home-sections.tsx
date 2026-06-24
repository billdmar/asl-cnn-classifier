"use client";

/**
 * Client-side lazy loaders for the home page's heavy, below-the-fold sections.
 *
 * The webcam, upload, and metrics panels pull in large dependencies
 * (onnxruntime-web ~410KB, recharts ~340KB, @mediapipe, framer-motion). None of
 * them are part of the LCP content, so we code-split each into its own chunk via
 * next/dynamic with { ssr: false } and a fixed-height skeleton. This keeps that
 * JS out of the initial bundle — it is fetched on demand after first paint
 * instead of blocking it.
 *
 * These wrappers live in a "use client" module so the home page (app/page.tsx)
 * can stay a Server Component while still using ssr:false dynamic imports.
 */

import dynamic from "next/dynamic";

import { SkeletonCard } from "@/components/skeleton-card";

export const WebcamPanel = dynamic(
  () => import("@/components/webcam/webcam-panel").then((m) => m.WebcamPanel),
  {
    ssr: false,
    loading: () => <SkeletonCard minHeight={420} label="Loading live demo" />,
  },
);

export const UploadPanel = dynamic(
  () => import("@/components/upload/upload-panel").then((m) => m.UploadPanel),
  {
    ssr: false,
    loading: () => <SkeletonCard minHeight={360} label="Loading upload panel" />,
  },
);

export const MetricsDashboard = dynamic(
  () => import("@/components/metrics/metrics-dashboard").then((m) => m.MetricsDashboard),
  {
    ssr: false,
    loading: () => <SkeletonCard minHeight={600} label="Loading metrics" />,
  },
);
