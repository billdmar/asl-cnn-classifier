import { test, expect } from "@playwright/test";

/**
 * PR-1 smoke E2E: the static site loads, renders the dark themed hero, exposes
 * the navigation landmarks, and serves the real ONNX model asset. Feature-level
 * E2E (upload a known image, assert the predicted letter) lands with the
 * features PR; this guards the foundation.
 */

test.describe("landing page", () => {
  test("renders the hero and core landmarks", async ({ page }) => {
    await page.goto("/");

    await expect(page).toHaveTitle(/ASL Classifier/);

    // Exactly one h1, and it is the hero headline.
    const h1 = page.getByRole("heading", { level: 1 });
    await expect(h1).toHaveText(/Read sign language in your browser/i);

    // Landmarks present for accessibility.
    await expect(page.getByRole("banner")).toBeVisible();
    await expect(page.getByRole("navigation", { name: /primary/i })).toBeVisible();
    await expect(page.getByRole("contentinfo")).toBeVisible();

    // Dark theme actually applied (not a flash of unstyled / white page).
    await page.evaluate(() =>
      document.documentElement.setAttribute("data-theme", "dark"),
    );
    const darkBg = await page.evaluate(
      () => getComputedStyle(document.body).backgroundColor,
    );
    expect(darkBg).toBe("rgb(10, 10, 15)");

    // Light theme repaints the page bg white (proves the palette swap is live).
    await page.evaluate(() =>
      document.documentElement.setAttribute("data-theme", "light"),
    );
    const lightBg = await page.evaluate(
      () => getComputedStyle(document.body).backgroundColor,
    );
    expect(lightBg).toBe("rgb(255, 255, 255)");
  });

  test("nav anchors resolve to on-page sections", async ({ page }) => {
    await page.goto("/");
    for (const id of ["live", "upload", "metrics", "how"]) {
      await expect(page.locator(`#${id}`)).toBeAttached();
    }
  });

  test("footer shows build provenance", async ({ page }) => {
    await page.goto("/");
    // Baked in at build time by next.config.mjs: "build <sha|dev>".
    await expect(page.getByRole("contentinfo")).toContainText(/build\s+\S+/i);
  });

  test("serves the real ONNX model asset", async ({ request }) => {
    const res = await request.get("/model/model.onnx");
    expect(res.status()).toBe(200);
    // The real 26-class MobileNetV2 export is ~9 MB; guard against a 404 page
    // or an empty/placeholder file silently shipping.
    const len = Number(res.headers()["content-length"] ?? "0");
    expect(len).toBeGreaterThan(1_000_000);
  });
});
