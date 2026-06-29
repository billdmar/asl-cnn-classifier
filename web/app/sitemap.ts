import type { MetadataRoute } from "next";

export const dynamic = "force-static";

const BASE = "https://asl-cnn-classifier.vercel.app";

export default function sitemap(): MetadataRoute.Sitemap {
  const lastModified = new Date();
  return [
    {
      url: `${BASE}/`,
      lastModified,
      changeFrequency: "monthly",
      priority: 1.0,
    },
    {
      url: `${BASE}/about`,
      lastModified,
      changeFrequency: "monthly",
      priority: 0.8,
    },
    {
      url: `${BASE}/result`,
      lastModified,
      changeFrequency: "monthly",
      priority: 0.5,
    },
  ];
}
