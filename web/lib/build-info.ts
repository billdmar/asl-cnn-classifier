/**
 * Build/deploy provenance shown in the footer so visitors (and the author) can
 * see exactly which commit is live. The raw values are baked into the static
 * export at build time via `next.config.mjs` env (NEXT_PUBLIC_BUILD_SHA /
 * NEXT_PUBLIC_BUILD_DATE); this module is the pure, testable formatter.
 */

const REPO_URL = "https://github.com/billdmar/asl-cnn-classifier";

/** Sentinel used when no real commit SHA is available (local dev). */
export const DEV_SHA = "dev";

export interface BuildInfo {
  /** Short (≤7 char) commit SHA, or "dev" / "" when unknown. */
  shortSha: string;
  /** GitHub commit URL, or null when the SHA is unknown (no link rendered). */
  commitUrl: string | null;
  /** Human-readable build date (e.g. "Jun 29, 2026"), or null when unknown. */
  date: string | null;
}

/**
 * Normalize the raw build env values into render-ready footer fields.
 *
 * - A real SHA is truncated to 7 chars and linked to its GitHub commit.
 * - The "dev" sentinel (or anything falsy) yields no commit link.
 * - An invalid/missing date yields `date: null` (the footer omits it) rather
 *   than rendering "Invalid Date".
 */
export function formatBuildInfo(
  rawSha: string | undefined,
  rawDate: string | undefined,
): BuildInfo {
  const sha = (rawSha ?? "").trim();
  const isReal = sha.length > 0 && sha !== DEV_SHA;
  const shortSha = isReal ? sha.slice(0, 7) : DEV_SHA;
  const commitUrl = isReal ? `${REPO_URL}/commit/${sha}` : null;

  let date: string | null = null;
  if (rawDate) {
    const parsed = new Date(rawDate);
    if (!Number.isNaN(parsed.getTime())) {
      date = parsed.toLocaleDateString("en-US", {
        year: "numeric",
        month: "short",
        day: "numeric",
      });
    }
  }

  return { shortSha, commitUrl, date };
}
