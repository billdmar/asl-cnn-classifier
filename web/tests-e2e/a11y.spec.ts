import AxeBuilder from "@axe-core/playwright";
import { test, expect } from "@playwright/test";

/**
 * Accessibility smoke: no serious/critical axe violations on the main routes.
 * Scoped to serious+critical so the gate is meaningful but not flaky on minor
 * contrast nits in third-party chart SVGs. Webcam/canvas surfaces are inert
 * until the user opts in, so the landing render is a fair a11y target.
 */
interface ThemedViolation {
  theme: string;
  id: string;
  help: string;
}

/**
 * Scan a route under BOTH themes and collect serious/critical axe violations,
 * tagging each with the theme so a light-only contrast regression is legible in
 * the failure message. Looping light here is the key gate proving the light
 * palette meets contrast.
 */
async function scan(page: import("@playwright/test").Page, url: string) {
  await page.goto(url);
  const found: ThemedViolation[] = [];
  for (const theme of ["dark", "light"] as const) {
    await page.evaluate(
      (t) => document.documentElement.setAttribute("data-theme", t),
      theme,
    );
    // Components carry `transition-colors`, so a programmatic theme swap leaves
    // colors mid-interpolation for a few hundred ms. WCAG contrast applies to
    // the settled state — wait for transitions to finish before sampling so axe
    // doesn't read a transient blended color (a real toggle fades the same way).
    await page.waitForTimeout(500);
    const results = await new AxeBuilder({ page })
      .withTags(["wcag2a", "wcag2aa"])
      .analyze();
    for (const v of results.violations) {
      if (v.impact === "serious" || v.impact === "critical") {
        found.push({ theme, id: v.id, help: v.help });
      }
    }
  }
  return found;
}

test.describe("accessibility", () => {
  test("home page has no serious/critical axe violations", async ({ page }) => {
    const violations = await scan(page, "/");
    expect(
      violations,
      violations.map((v) => `[${v.theme}] ${v.id}: ${v.help}`).join("\n"),
    ).toEqual([]);
  });

  test("about page has no serious/critical axe violations", async ({ page }) => {
    const violations = await scan(page, "/about");
    expect(
      violations,
      violations.map((v) => `[${v.theme}] ${v.id}: ${v.help}`).join("\n"),
    ).toEqual([]);
  });

  test("result page (empty state) has no serious/critical axe violations", async ({
    page,
  }) => {
    const violations = await scan(page, "/result");
    expect(
      violations,
      violations.map((v) => `[${v.theme}] ${v.id}: ${v.help}`).join("\n"),
    ).toEqual([]);
  });
});
