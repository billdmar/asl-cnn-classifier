import AxeBuilder from "@axe-core/playwright";
import { test, expect } from "@playwright/test";

/**
 * Accessibility smoke: no serious/critical axe violations on the main routes.
 * Scoped to serious+critical so the gate is meaningful but not flaky on minor
 * contrast nits in third-party chart SVGs. Webcam/canvas surfaces are inert
 * until the user opts in, so the landing render is a fair a11y target.
 */
async function scan(page: import("@playwright/test").Page, url: string) {
  await page.goto(url);
  const results = await new AxeBuilder({ page })
    .withTags(["wcag2a", "wcag2aa"])
    .analyze();
  const serious = results.violations.filter(
    (v) => v.impact === "serious" || v.impact === "critical",
  );
  return serious;
}

test.describe("accessibility", () => {
  test("home page has no serious/critical axe violations", async ({ page }) => {
    const violations = await scan(page, "/");
    expect(
      violations,
      violations.map((v) => `${v.id}: ${v.help}`).join("\n"),
    ).toEqual([]);
  });

  test("about page has no serious/critical axe violations", async ({ page }) => {
    const violations = await scan(page, "/about");
    expect(
      violations,
      violations.map((v) => `${v.id}: ${v.help}`).join("\n"),
    ).toEqual([]);
  });
});
