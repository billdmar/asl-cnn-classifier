import { describe, expect, it } from "vitest";

import { buildJsonLd } from "@/lib/structured-data";

describe("buildJsonLd", () => {
  it("has the correct schema.org context and type", () => {
    const jsonLd = buildJsonLd();
    expect(jsonLd["@context"]).toBe("https://schema.org");
    expect(jsonLd["@type"]).toBe("WebApplication");
  });

  it("has a non-empty name, url, and description", () => {
    const jsonLd = buildJsonLd();
    expect(jsonLd.name).toBe("ASL Classifier");
    expect(typeof jsonLd.url).toBe("string");
    expect((jsonLd.url as string).length).toBeGreaterThan(0);
    expect(typeof jsonLd.description).toBe("string");
    expect((jsonLd.description as string).length).toBeGreaterThan(0);
  });

  it("uses the live cross-dataset accuracy figure in the description", () => {
    const jsonLd = buildJsonLd();
    expect(jsonLd.description as string).toContain("59.8%");
  });

  it("serializes cleanly with no undefined/functions for the inline <script>", () => {
    const serialized = JSON.stringify(buildJsonLd());
    expect(serialized).not.toContain("undefined");
    // Round-trips back to an equivalent object.
    expect(JSON.parse(serialized)).toEqual(buildJsonLd());
  });

  it("exposes a non-empty featureList array", () => {
    const jsonLd = buildJsonLd();
    expect(Array.isArray(jsonLd.featureList)).toBe(true);
    expect((jsonLd.featureList as unknown[]).length).toBeGreaterThan(0);
  });
});
