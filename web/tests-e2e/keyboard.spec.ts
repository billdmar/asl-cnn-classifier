import AxeBuilder from "@axe-core/playwright";
import { test, expect } from "@playwright/test";

/**
 * Keyboard shortcuts + help dialog. The native <dialog> opens via the "?"
 * shortcut (and the header button), traps focus, and must stay axe-clean while
 * open. We re-run the serious/critical axe scan WITH THE DIALOG OPEN — the
 * landing scans in a11y.spec.ts only see the closed (absent) state.
 */
test.describe("keyboard help dialog", () => {
  test("opens via '?', is axe-clean, and Escape closes it", async ({
    page,
  }) => {
    await page.goto("/");

    await page.keyboard.press("?");

    const dialog = page.getByRole("dialog", { name: /keyboard shortcuts/i });
    await expect(dialog).toBeVisible({ timeout: 5000 });
    await expect(
      dialog.getByRole("heading", { name: /keyboard shortcuts/i }),
    ).toBeVisible();

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

    await page.keyboard.press("Escape");
    await expect(dialog).toBeHidden({ timeout: 5000 });

    // Focus returns to the document body / page after a native dialog closes;
    // the trigger button remains reachable and the page is still interactive.
    await expect(
      page.getByRole("button", { name: /keyboard shortcuts/i }),
    ).toBeVisible();
  });

  test("opens from the header '?' button", async ({ page }) => {
    await page.goto("/");

    await page
      .getByRole("button", { name: /keyboard shortcuts/i })
      .click();

    await expect(
      page.getByRole("dialog", { name: /keyboard shortcuts/i }),
    ).toBeVisible({ timeout: 5000 });
  });
});
