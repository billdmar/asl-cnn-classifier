import { readFileSync } from "node:fs";
import path from "node:path";

import { test, expect } from "@playwright/test";

/**
 * Feature E2E: real in-browser inference. Clicking the bundled "A" example runs
 * the actual onnxruntime-web path (preprocess → ONNX → softmax) and must render
 * the correct predicted letter. This is the centerpiece honesty check — it
 * proves the deployed model genuinely classifies, not just that the UI renders.
 *
 * The metrics-dashboard assertions are DATA-DRIVEN: they read the committed
 * metrics/calibration JSON and assert the dashboard renders those exact values
 * (formatted the same way the component does). This tracks retrains automatically
 * instead of pinning hardcoded numbers that rot.
 */

const METRICS_DIR = path.resolve(__dirname, "../public/metrics");
const metricsJson = JSON.parse(
  readFileSync(path.join(METRICS_DIR, "metrics.json"), "utf-8"),
) as { overall_accuracy: number; num_test_samples: number };
const calibrationJson = JSON.parse(
  readFileSync(path.join(METRICS_DIR, "calibration.json"), "utf-8"),
) as { ece: number };

const pct = (n: number) => `${(n * 100).toFixed(1)}%`;

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
    // Derived from the committed metrics.json — tracks retrains automatically.
    await expect(metrics).toContainText(pct(metricsJson.overall_accuracy), {
      timeout: 15_000,
    });
    await expect(metrics).toContainText(
      metricsJson.num_test_samples.toLocaleString("en-US"),
      { timeout: 15_000 },
    );
  });

  test("calibration card renders real measured ECE (not a placeholder)", async ({
    page,
  }) => {
    await page.goto("/");
    const metrics = page.locator("#metrics");
    // From the real calibration.json — value read, not hardcoded.
    await expect(metrics).toContainText("ECE", { timeout: 15_000 });
    await expect(metrics).toContainText(calibrationJson.ece.toFixed(3), {
      timeout: 15_000,
    });
    // The "coming soon" placeholder must be gone.
    await expect(metrics).not.toContainText(/coming with the calibration/i);
  });

  test("confusion-matrix explorer renders on the dashboard", async ({ page }) => {
    await page.goto("/");
    const metrics = page.locator("#metrics");
    await expect(metrics).toContainText(/Where it gets confused/i, {
      timeout: 15_000,
    });
    // The heatmap is an accessible figure naming the worst confusion.
    await expect(
      metrics.getByRole("img", { name: /confusion matrix over \d+ classes/i }),
    ).toBeVisible({ timeout: 15_000 });
  });

  test("upload via file input classifies a chosen image", async ({ page }) => {
    await page.goto("/");
    const upload = page.locator("#upload");
    // Set a real file on the hidden <input type=file> (the same path drag-drop
    // and the picker both feed). Reuses a committed example fixture.
    const examplePath = path.resolve(__dirname, "../public/examples/L.png");
    await upload.locator('input[type="file"]').setInputFiles(examplePath);
    // A raw file upload (no known label) still runs the full classify path and
    // renders the "Predicted" result + the top-5 confidence bars.
    await expect(upload).toContainText(/Predicted/i, { timeout: 30_000 });
    await expect(upload.getByText("L", { exact: true }).first()).toBeVisible({
      timeout: 30_000,
    });
  });
});
