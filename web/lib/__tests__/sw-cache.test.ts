/**
 * Unit tests for the service-worker caching policy (lib/sw-cache.ts) plus a
 * PARITY gate asserting public/sw.js still mirrors the same path literals.
 *
 * The SW is hand-written plain JS (it can't import this TS module), so the
 * policy lives in two places; this parity check (mirroring the parity.*.test.ts
 * idiom of reading a sibling file with fs) prevents the copies from drifting.
 */

import { describe, it, expect } from "vitest";
import path from "node:path";
import { readFileSync } from "node:fs";

import { cacheName, shouldCache } from "../sw-cache";

const SELF_ORIGIN = "https://asl.example.com";

function u(pathname: string, origin = SELF_ORIGIN): URL {
  return new URL(pathname, origin);
}

describe("shouldCache", () => {
  it("never caches the IndexedDB-owned model path (the critical guard)", () => {
    expect(shouldCache(u("/model/model.onnx"), SELF_ORIGIN)).toBe(false);
  });

  it("never caches live metrics JSON", () => {
    expect(shouldCache(u("/metrics/x.json"), SELF_ORIGIN)).toBe(false);
  });

  it("excludes mediapipe (handled by a dedicated cache-first path)", () => {
    expect(shouldCache(u("/mediapipe/hand_landmarker.task"), SELF_ORIGIN)).toBe(
      false,
    );
  });

  it("caches HTML and same-origin static assets", () => {
    expect(shouldCache(u("/index.html"), SELF_ORIGIN)).toBe(true);
    expect(shouldCache(u("/_next/static/chunk.js"), SELF_ORIGIN)).toBe(true);
  });

  it("never caches cross-origin requests (e.g. the ORT CDN)", () => {
    expect(
      shouldCache(
        new URL("https://cdn.jsdelivr.net/npm/onnxruntime-web/ort.wasm"),
        "https://a.com",
      ),
    ).toBe(false);
  });
});

describe("cacheName", () => {
  it("builds the versioned cache name", () => {
    expect(cacheName("abc")).toBe("asl-cache-abc");
  });
});

describe("parity: public/sw.js mirrors the policy literals", () => {
  const sw = readFileSync(
    path.join(__dirname, "..", "..", "public", "sw.js"),
    "utf-8",
  );

  it("contains the never-cache path prefixes", () => {
    expect(sw).toContain("/model/");
    expect(sw).toContain("/mediapipe/");
    expect(sw).toContain("/metrics/");
  });
});
