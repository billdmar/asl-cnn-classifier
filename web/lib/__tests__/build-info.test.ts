/**
 * Unit tests for the pure build-info formatter that powers the footer's deploy
 * provenance line.
 */

import { describe, it, expect } from "vitest";

import { formatBuildInfo, DEV_SHA } from "../build-info";

const REPO = "https://github.com/billdmar/asl-cnn-classifier";

describe("formatBuildInfo", () => {
  it("truncates a real SHA to 7 chars and links to the GitHub commit", () => {
    const sha = "34e59328e39b7a5aa3f8cb2f099e2ed9f78e97ec";
    const info = formatBuildInfo(sha, "2026-06-29T16:20:55Z");
    expect(info.shortSha).toBe("34e5932");
    expect(info.commitUrl).toBe(`${REPO}/commit/${sha}`);
    expect(info.date).toBe("Jun 29, 2026");
  });

  it("falls back to the dev sentinel with no commit link when SHA is missing", () => {
    const info = formatBuildInfo(undefined, undefined);
    expect(info.shortSha).toBe(DEV_SHA);
    expect(info.commitUrl).toBeNull();
    expect(info.date).toBeNull();
  });

  it("treats the literal 'dev' SHA as no commit link", () => {
    const info = formatBuildInfo("dev", undefined);
    expect(info.shortSha).toBe(DEV_SHA);
    expect(info.commitUrl).toBeNull();
  });

  it("trims whitespace and ignores an empty SHA", () => {
    expect(formatBuildInfo("   ", undefined).commitUrl).toBeNull();
    const info = formatBuildInfo("  abc1234def  ", undefined);
    expect(info.shortSha).toBe("abc1234");
  });

  it("returns null date (not 'Invalid Date') for an unparseable date", () => {
    expect(formatBuildInfo("abc1234", "not-a-date").date).toBeNull();
  });
});
