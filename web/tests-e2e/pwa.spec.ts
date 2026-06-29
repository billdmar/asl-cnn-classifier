import { test, expect } from "@playwright/test";

/**
 * PWA / installability + SEO surface. The service worker is registration-guarded
 * to skip localhost (so the inference/a11y/smoke specs run on a clean SW-free
 * origin — see components/service-worker-register.tsx), so this spec does NOT
 * test the SW at runtime; it verifies the deterministic, file-backed pieces:
 * the manifest is linked + valid + its icons exist, and the JSON-LD is present
 * and parseable. `request.get` bypasses any SW (mirrors smoke.spec's ONNX check).
 */

test.describe("pwa + seo", () => {
  test("manifest is linked, valid, and its icons resolve", async ({
    page,
    request,
  }) => {
    await page.goto("/");

    const href = await page
      .locator('link[rel="manifest"]')
      .getAttribute("href");
    expect(href, "a <link rel=manifest> is present").toBeTruthy();

    const res = await request.get(href!);
    expect(res.status()).toBe(200);
    const manifest = (await res.json()) as {
      display?: string;
      start_url?: string;
      icons?: { src: string; sizes: string; purpose?: string }[];
    };

    expect(manifest.display).toBe("standalone");
    expect(manifest.start_url).toBeTruthy();
    const icons = manifest.icons ?? [];
    expect(icons.some((i) => i.sizes === "192x192")).toBe(true);
    expect(icons.some((i) => i.sizes === "512x512")).toBe(true);
    expect(icons.some((i) => i.purpose === "maskable")).toBe(true);

    // Every referenced icon must actually exist (no 404 in the install prompt).
    for (const icon of icons) {
      const iconRes = await request.get(icon.src);
      expect(iconRes.status(), `icon ${icon.src} resolves`).toBe(200);
    }
  });

  test("structured data (JSON-LD) is present and parseable", async ({ page }) => {
    await page.goto("/");
    const raw = await page
      .locator('script[type="application/ld+json"]')
      .textContent();
    expect(raw, "JSON-LD script present").toBeTruthy();
    const data = JSON.parse(raw!) as { "@type"?: string; description?: string };
    expect(data["@type"]).toBe("WebApplication");
    // Honest accuracy stays in the structured data, not a fake user rating.
    expect(data.description).toContain("59.8%");
  });

  test("sitemap and robots are served", async ({ request }) => {
    expect((await request.get("/sitemap.xml")).status()).toBe(200);
    const robots = await request.get("/robots.txt");
    expect(robots.status()).toBe(200);
    expect(await robots.text()).toContain("Sitemap");
  });
});
