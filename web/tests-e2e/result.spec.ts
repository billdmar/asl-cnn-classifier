import AxeBuilder from "@axe-core/playwright";
import { test, expect } from "@playwright/test";

/**
 * Shared-result permalink: a known payload decodes and renders, and the page
 * has no serious/critical axe violations.
 *
 * The fixture is `{letter:"A",topk:[["A",0.99]],t:0,v:1}` base64url-encoded the
 * same way lib/share-link.ts encodes (JSON -> UTF-8 -> base64url, no padding).
 */
const FIXTURE = "eyJsZXR0ZXIiOiJBIiwidG9wayI6W1siQSIsMC45OV1dLCJ0IjowLCJ2IjoxfQ";

test.describe("shared result permalink", () => {
  test("renders the decoded letter and confidence", async ({ page }) => {
    await page.goto(`/result#r=${FIXTURE}`);
    // The big predicted letter.
    await expect(
      page.getByText("A", { exact: true }).first(),
    ).toBeVisible({ timeout: 15_000 });
    // The top-k bar percentage.
    await expect(page.getByText("99.0%")).toBeVisible({ timeout: 15_000 });
  });

  test("shows a friendly empty state for an invalid hash", async ({ page }) => {
    await page.goto("/result#r=garbage!!!");
    await expect(page.getByText(/no result to show/i)).toBeVisible({
      timeout: 15_000,
    });
  });

  test("has no serious/critical axe violations", async ({ page }) => {
    await page.goto(`/result#r=${FIXTURE}`);
    await expect(page.getByText("99.0%")).toBeVisible({ timeout: 15_000 });
    const results = await new AxeBuilder({ page })
      .withTags(["wcag2a", "wcag2aa"])
      .analyze();
    const serious = results.violations.filter(
      (v) => v.impact === "serious" || v.impact === "critical",
    );
    expect(
      serious,
      serious.map((v) => `${v.id}: ${v.help}`).join("\n"),
    ).toEqual([]);
  });
});
