import { describe, expect, it } from "vitest";

import { decodeResult, encodeResult, type SharedResult } from "@/lib/share-link";
import type { InferenceResult, Prediction } from "@/lib/inference";

/** Build a minimal InferenceResult from `[label, prob]` pairs (already ranked). */
function makeResult(pairs: [string, number][]): InferenceResult {
  const ranked: Prediction[] = pairs.map(([label, prob], index) => ({
    label,
    index,
    prob,
  }));
  const probs = new Float32Array(ranked.map((p) => p.prob));
  return { ranked, probs, top: ranked[0]! };
}

/** base64url-encode an arbitrary object the same way the encoder does. */
function b64url(obj: unknown): string {
  const bytes = new TextEncoder().encode(JSON.stringify(obj));
  let binary = "";
  for (const b of bytes) binary += String.fromCharCode(b);
  return btoa(binary).replace(/\+/g, "-").replace(/\//g, "_").replace(/=+$/, "");
}

describe("encodeResult / decodeResult round-trip", () => {
  it("round-trips a typical result with a fixed timestamp", () => {
    const result = makeResult([
      ["A", 0.987654],
      ["S", 0.008],
      ["E", 0.003],
    ]);
    const encoded = encodeResult(result, 1_700_000_000_000);
    const decoded = decodeResult(encoded);

    expect(decoded).toEqual<SharedResult>({
      letter: "A",
      topk: [
        ["A", 0.9877], // rounded to 4 dp
        ["S", 0.008],
        ["E", 0.003],
      ],
      t: 1_700_000_000_000,
      v: 1,
    });
  });

  it("caps top-k at 5 entries", () => {
    const result = makeResult([
      ["A", 0.5],
      ["B", 0.2],
      ["C", 0.1],
      ["D", 0.08],
      ["E", 0.07],
      ["F", 0.05],
    ]);
    const decoded = decodeResult(encodeResult(result, 0));
    expect(decoded?.topk).toHaveLength(5);
    expect(decoded?.topk.map((p) => p[0])).toEqual(["A", "B", "C", "D", "E"]);
  });

  it("is pure — same input + timestamp yields identical output", () => {
    const result = makeResult([["A", 0.9]]);
    expect(encodeResult(result, 42)).toBe(encodeResult(result, 42));
  });
});

describe("decodeResult rejection (returns null, never throws)", () => {
  it("rejects malformed base64", () => {
    expect(decodeResult("!!!not base64!!!")).toBeNull();
  });

  it("rejects a payload that isn't JSON", () => {
    const raw = btoa("not json at all").replace(/=+$/, "");
    expect(decodeResult(raw)).toBeNull();
  });

  it("rejects the empty string", () => {
    expect(decodeResult("")).toBeNull();
  });

  it("rejects oversized input (>2KB)", () => {
    expect(decodeResult("A".repeat(2049))).toBeNull();
  });

  it("rejects the wrong schema version", () => {
    expect(decodeResult(b64url({ letter: "A", topk: [], t: 0, v: 2 }))).toBeNull();
  });

  it("rejects a non-numeric / non-finite timestamp", () => {
    expect(decodeResult(b64url({ letter: "A", topk: [], t: "soon", v: 1 }))).toBeNull();
    expect(decodeResult(b64url({ letter: "A", topk: [], t: null, v: 1 }))).toBeNull();
  });

  it("rejects an empty or oversized letter", () => {
    expect(decodeResult(b64url({ letter: "", topk: [], t: 0, v: 1 }))).toBeNull();
    expect(
      decodeResult(b64url({ letter: "x".repeat(25), topk: [], t: 0, v: 1 })),
    ).toBeNull();
  });

  it("rejects topk that isn't an array", () => {
    expect(decodeResult(b64url({ letter: "A", topk: "nope", t: 0, v: 1 }))).toBeNull();
  });

  it("rejects topk longer than 5", () => {
    const topk = Array.from({ length: 6 }, (_, i) => [`L${i}`, 0.1]);
    expect(decodeResult(b64url({ letter: "A", topk, t: 0, v: 1 }))).toBeNull();
  });

  it("rejects bad topk tuple shapes", () => {
    expect(
      decodeResult(b64url({ letter: "A", topk: [["A"]], t: 0, v: 1 })),
    ).toBeNull();
    expect(
      decodeResult(b64url({ letter: "A", topk: [[1, 0.5]], t: 0, v: 1 })),
    ).toBeNull();
    expect(
      decodeResult(b64url({ letter: "A", topk: [["A", "high"]], t: 0, v: 1 })),
    ).toBeNull();
  });

  it("clamps out-of-range probabilities into [0,1]", () => {
    const decoded = decodeResult(
      b64url({ letter: "A", topk: [["A", 1.7], ["B", -0.4]], t: 0, v: 1 }),
    );
    expect(decoded?.topk).toEqual([
      ["A", 1],
      ["B", 0],
    ]);
  });

  it("rejects non-finite probabilities outright", () => {
    // JSON can't carry Infinity, but a NaN-producing value would arrive as null.
    expect(
      decodeResult(b64url({ letter: "A", topk: [["A", null]], t: 0, v: 1 })),
    ).toBeNull();
  });

  it("rejects a non-object payload", () => {
    expect(decodeResult(b64url(42))).toBeNull();
    expect(decodeResult(b64url("just a string"))).toBeNull();
    expect(decodeResult(b64url(null))).toBeNull();
  });
});
