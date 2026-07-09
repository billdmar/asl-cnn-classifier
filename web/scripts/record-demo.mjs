/**
 * Automated demo recording — captures the inference explainer flow as a video
 * for the README's animated hero.
 *
 * Drives the static production build with Playwright, records the browser,
 * and saves a .webm to docs/. Convert to GIF with:
 *
 *   ffmpeg -i docs/demo.webm -vf "fps=12,scale=800:-1:flags=lanczos" \
 *     -loop 0 docs/demo.gif
 *
 * Usage (from web/):
 *   npm run build                 # produce the static export in out/
 *   node scripts/record-demo.mjs  # serves out/, records, writes ../docs/demo.webm
 */

import { chromium } from "@playwright/test";
import { createServer } from "node:http";
import { readFile, rename, readdir } from "node:fs/promises";
import { existsSync } from "node:fs";
import { extname, join } from "node:path";

const PORT = 4322;
const OUT_DIR = "out";
const VIDEO_DIR = "../docs";
const MIME = {
  ".html": "text/html",
  ".js": "text/javascript",
  ".mjs": "text/javascript",
  ".css": "text/css",
  ".json": "application/json",
  ".png": "image/png",
  ".svg": "image/svg+xml",
  ".ico": "image/x-icon",
  ".onnx": "application/octet-stream",
  ".wasm": "application/wasm",
  ".task": "application/octet-stream",
  ".webmanifest": "application/manifest+json",
  ".txt": "text/plain",
  ".xml": "application/xml",
};

if (!existsSync(OUT_DIR)) {
  console.error("No out/ directory — run `npm run build` first.");
  process.exit(1);
}

// Minimal static file server for the export.
const server = createServer(async (req, res) => {
  try {
    let path = (req.url ?? "/").split("?")[0];
    if (path.endsWith("/")) path += "index.html";
    let file = join(OUT_DIR, path);
    if (!existsSync(file) && existsSync(`${file}.html`)) file = `${file}.html`;
    const body = await readFile(file);
    res.writeHead(200, {
      "Content-Type": MIME[extname(file)] ?? "application/octet-stream",
    });
    res.end(body);
  } catch {
    res.writeHead(404);
    res.end("not found");
  }
});

await new Promise((resolve) => server.listen(PORT, resolve));
console.log(`Serving ${OUT_DIR}/ on http://localhost:${PORT}`);

const browser = await chromium.launch();
const context = await browser.newContext({
  viewport: { width: 1280, height: 800 },
  recordVideo: { dir: VIDEO_DIR, size: { width: 1280, height: 800 } },
  colorScheme: "dark",
});
const page = await context.newPage();

try {
  console.log("Loading page…");
  await page.goto(`http://localhost:${PORT}/`, { waitUntil: "networkidle" });
  await page.waitForTimeout(1500);

  // Scroll through the hero for context.
  console.log("Scrolling to explainer…");
  const explainer = page.locator("#explainer");
  await explainer.scrollIntoViewIfNeeded();
  await page.waitForTimeout(1200);

  // Run the pipeline on the letter A example.
  console.log("Running the pipeline on example A…");
  await explainer.getByRole("button", { name: /Explain letter A/i }).click();
  await explainer
    .getByRole("tablist", { name: /Pipeline steps/i })
    .waitFor({ state: "visible", timeout: 60_000 });
  await page.waitForTimeout(1500);

  // Step through all 5 stages with a beat on each.
  for (let step = 2; step <= 5; step++) {
    console.log(`Step ${step}…`);
    await explainer.getByRole("tab", { name: new RegExp(`Step ${step}`) }).click();
    await page.waitForTimeout(1600);
  }

  // Drag the temperature slider to show the distribution reshape.
  console.log("Dragging the temperature slider…");
  const slider = explainer.getByRole("slider", { name: /Temperature/i });
  await slider.waitFor({ state: "visible" });
  for (const value of ["2.0", "3.0", "0.6", "1.0"]) {
    await slider.fill(value);
    await page.waitForTimeout(900);
  }
  await page.waitForTimeout(1000);

  console.log("Recording complete.");
} finally {
  await context.close(); // flushes the video file
  await browser.close();
  server.close();
}

// Rename the auto-named video to docs/demo.webm.
const files = await readdir(VIDEO_DIR);
const video = files.filter((f) => f.endsWith(".webm")).sort().pop();
if (video && video !== "demo.webm") {
  await rename(join(VIDEO_DIR, video), join(VIDEO_DIR, "demo.webm"));
}
console.log(`Saved ${VIDEO_DIR}/demo.webm`);
console.log(
  'Convert to GIF: ffmpeg -i docs/demo.webm -vf "fps=12,scale=800:-1:flags=lanczos" -loop 0 docs/demo.gif',
);
