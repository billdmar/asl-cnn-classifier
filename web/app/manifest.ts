import type { MetadataRoute } from "next";
import { manifest } from "@/lib/manifest";

// Static-export friendly: force-static makes Next pre-render the manifest to a
// single static file at build time (required under `output: export`). Next
// auto-injects the <link rel="manifest"> into <head>.
export const dynamic = "force-static";

export default function manifestRoute(): MetadataRoute.Manifest {
  return manifest;
}
