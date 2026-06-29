/**
 * Shareable result permalinks (pure, no React, static-export safe).
 *
 * A classification result is encoded into a compact, URL-safe payload that
 * lives entirely in the URL hash fragment (`/result#r=<payload>`). Because the
 * site is a static export with no backend, the hash is the storage: it never
 * touches a server, and `/result` decodes it client-side. The encoder is kept
 * deliberately PURE — the timestamp is passed in, never read from `Date.now()`
 * here — so round-trips are deterministic and unit-testable.
 */

import type { InferenceResult } from "./inference";

/** The decoded shape carried in a share link. `v` is the schema version. */
export interface SharedResult {
  /** Top-1 predicted label. */
  letter: string;
  /** Top-k `[label, prob]` pairs (prob in [0,1]), at most 5, ranked desc. */
  topk: [string, number][];
  /** Unix epoch milliseconds the share was created. */
  t: number;
  /** Schema version literal. */
  v: 1;
}

/** Max number of top-k entries carried in a link. */
const MAX_TOPK = 5;
/** Max label length we accept (guards against junk payloads). */
const MAX_LABEL_LEN = 24;
/** Hard cap on the raw encoded string we will even attempt to decode (~2 KB). */
const MAX_RAW_LEN = 2048;

/** Round a number to `digits` decimal places. */
function round(value: number, digits: number): number {
  const f = 10 ** digits;
  return Math.round(value * f) / f;
}

/** base64url-encode a UTF-8 string (URL-safe alphabet, no padding). */
function toBase64Url(json: string): string {
  // Encode as UTF-8 bytes first so multibyte labels survive btoa.
  const bytes = new TextEncoder().encode(json);
  let binary = "";
  for (const b of bytes) binary += String.fromCharCode(b);
  return btoa(binary).replace(/\+/g, "-").replace(/\//g, "_").replace(/=+$/, "");
}

/** Decode a base64url string back to its UTF-8 source, or null on failure. */
function fromBase64Url(raw: string): string | null {
  try {
    const b64 = raw.replace(/-/g, "+").replace(/_/g, "/");
    const binary = atob(b64);
    const bytes = new Uint8Array(binary.length);
    for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i);
    return new TextDecoder().decode(bytes);
  } catch {
    return null;
  }
}

/**
 * Encode an inference result into a URL-safe payload string.
 *
 * Pure: pass the timestamp in (`Date.now()` at the call site) rather than
 * reading the clock here, so the function is deterministic and testable.
 *
 * @param result - The inference result to share.
 * @param timestamp - Unix epoch ms to stamp the share with.
 * @returns A base64url string (no `#r=` prefix).
 */
export function encodeResult(result: InferenceResult, timestamp: number): string {
  const topk: [string, number][] = result.ranked
    .slice(0, MAX_TOPK)
    .map((p) => [p.label, round(p.prob, 4)]);
  const payload: SharedResult = {
    letter: result.top.label,
    topk,
    t: timestamp,
    v: 1,
  };
  return toBase64Url(JSON.stringify(payload));
}

/** Clamp a number into the closed interval [0, 1]. */
function clamp01(n: number): number {
  if (n < 0) return 0;
  if (n > 1) return 1;
  return n;
}

/**
 * Decode and STRICTLY validate a share payload. Never throws; returns null on
 * any malformation, wrong version, or oversize input. Probabilities are clamped
 * into [0,1]; out-of-shape entries reject the whole payload.
 *
 * @param raw - The base64url payload (without `#r=`).
 * @returns The validated {@link SharedResult}, or null.
 */
export function decodeResult(raw: string): SharedResult | null {
  if (typeof raw !== "string" || raw.length === 0 || raw.length > MAX_RAW_LEN) {
    return null;
  }
  const json = fromBase64Url(raw);
  if (json === null) return null;

  let parsed: unknown;
  try {
    parsed = JSON.parse(json);
  } catch {
    return null;
  }
  if (parsed === null || typeof parsed !== "object") return null;
  const obj = parsed as Record<string, unknown>;

  if (obj["v"] !== 1) return null;

  const t = obj["t"];
  if (typeof t !== "number" || !Number.isFinite(t)) return null;

  const letter = obj["letter"];
  if (typeof letter !== "string" || letter.length === 0 || letter.length > MAX_LABEL_LEN) {
    return null;
  }

  const rawTopk = obj["topk"];
  if (!Array.isArray(rawTopk) || rawTopk.length > MAX_TOPK) return null;

  const topk: [string, number][] = [];
  for (const entry of rawTopk) {
    if (!Array.isArray(entry) || entry.length !== 2) return null;
    const label = entry[0];
    const prob = entry[1];
    if (typeof label !== "string" || label.length === 0 || label.length > MAX_LABEL_LEN) {
      return null;
    }
    if (typeof prob !== "number" || !Number.isFinite(prob)) return null;
    topk.push([label, clamp01(prob)]);
  }

  return { letter, topk, t, v: 1 };
}
