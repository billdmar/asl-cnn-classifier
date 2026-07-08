import { test, expect } from "@playwright/test";

test.describe("Inference explainer", () => {
  test.beforeEach(async ({ page }) => {
    await page.goto("/");
  });

  test("loads and shows example buttons", async ({ page }) => {
    const section = page.locator("#explainer");
    await section.scrollIntoViewIfNeeded();
    await expect(section.getByRole("button", { name: /Explain letter A/i })).toBeVisible();
  });

  test("clicking an example runs the pipeline and shows steps", async ({ page }) => {
    const section = page.locator("#explainer");
    await section.scrollIntoViewIfNeeded();
    await section.getByRole("button", { name: /Explain letter A/i }).click();
    // Wait for pipeline to complete — the stepper should appear
    await expect(section.getByRole("tablist", { name: /Pipeline steps/i })).toBeVisible({ timeout: 30000 });
    // All 5 step tabs should be present
    const tabs = section.getByRole("tab");
    await expect(tabs).toHaveCount(5);
  });

  test("step navigation works", async ({ page }) => {
    const section = page.locator("#explainer");
    await section.scrollIntoViewIfNeeded();
    await section.getByRole("button", { name: /Explain letter A/i }).click();
    await expect(section.getByRole("tablist")).toBeVisible({ timeout: 30000 });
    // Click step 3 (Crop)
    await section.getByRole("tab", { name: /Step 3/i }).click();
    // The crop step should show "128 × 128"
    await expect(section.getByText("128 × 128")).toBeVisible();
  });

  test("temperature slider is interactive", async ({ page }) => {
    const section = page.locator("#explainer");
    await section.scrollIntoViewIfNeeded();
    await section.getByRole("button", { name: /Explain letter A/i }).click();
    await expect(section.getByRole("tablist")).toBeVisible({ timeout: 30000 });
    // Navigate to step 5 (Prediction)
    await section.getByRole("tab", { name: /Step 5/i }).click();
    // Temperature slider should exist
    const slider = section.getByRole("slider", { name: /Temperature/i });
    await expect(slider).toBeVisible();
  });
});
