import { describe, it, expect } from "vitest";
import { manifest } from "@/lib/manifest";

describe("PWA manifest", () => {
  it("declares standalone display", () => {
    expect(manifest.display).toBe("standalone");
  });

  it("has a truthy start_url", () => {
    expect(manifest.start_url).toBeTruthy();
  });

  it("uses the dark brand theme color", () => {
    expect(manifest.theme_color).toBe("#0a0a0f");
  });

  it("includes a 192x192 and a 512x512 icon", () => {
    const sizes = (manifest.icons ?? []).map((i) => i.sizes);
    expect(sizes).toContain("192x192");
    expect(sizes).toContain("512x512");
  });

  it("has at least one maskable icon", () => {
    const purposes = (manifest.icons ?? []).map((i) => i.purpose);
    expect(purposes).toContain("maskable");
  });

  it("references every icon by absolute path", () => {
    for (const icon of manifest.icons ?? []) {
      expect(icon.src.startsWith("/")).toBe(true);
    }
  });
});
