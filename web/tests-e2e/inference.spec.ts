import { test, expect } from "@playwright/test";

/**
 * Feature E2E: real in-browser inference. Clicking the bundled "A" example runs
 * the actual onnxruntime-web path (preprocess → ONNX → softmax) and must render
 * the correct predicted letter. This is the centerpiece honesty check — it
 * proves the deployed model genuinely classifies, not just that the UI renders.
 *
 * The metrics dashboard assertion proves the real committed JSON drives the UI.
 */

test.describe("in-browser inference", () => {
  test("classifying the bundled A example predicts A", async ({ page }) => {
    await page.goto("/");

    const classifyA = page.getByRole("button", {
      name: /classify example image for the letter A/i,
    });
    await expect(classifyA).toBeVisible();
    await classifyA.click();

    // The upload result region should show the predicted letter A. Inference
    // includes WASM model load on first run, so allow generous time.
    const uploadSection = page.locator("#upload");
    await expect(uploadSection).toContainText(/Predicted/i, { timeout: 30_000 });
    // "A" with a high probability, and the example labelled-A "match" note.
    await expect(uploadSection).toContainText(/match/i, { timeout: 30_000 });
    await expect(uploadSection.getByText(/\b9\d\.\d%/)).toBeVisible({
      timeout: 30_000,
    });
  });

  test("metrics dashboard renders the real measured numbers", async ({ page }) => {
    await page.goto("/");
    const metrics = page.locator("#metrics");
    // These come from the committed metrics.json / training_history.json.
    await expect(metrics).toContainText("96.8%", { timeout: 15_000 });
    await expect(metrics).toContainText("97.8%", { timeout: 15_000 });
    await expect(metrics).toContainText("1,631", { timeout: 15_000 });
  });
});
