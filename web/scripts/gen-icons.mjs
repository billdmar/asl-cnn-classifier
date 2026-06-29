// Generates the PWA manifest PNG icons from the brand "A" glyph using `sharp`
// (already present transitively via Next — no new dependency).
//
// Outputs to web/public/icons/:
//   icon-192.png            192x192 "any" icon (rounded-square, edge-to-edge glyph)
//   icon-512.png            512x512 "any" icon
//   icon-512-maskable.png   512x512 maskable icon — full-bleed #0a0a0f field with
//                           the glyph at ~60% so circular launcher masks don't clip it.
//
// Run once, then commit the PNGs. NOT wired into prebuild (keeps prod build dep-free):
//   node scripts/gen-icons.mjs

import sharp from "sharp";
import { mkdir } from "node:fs/promises";
import { fileURLToPath } from "node:url";
import path from "node:path";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const OUT_DIR = path.resolve(__dirname, "../public/icons");

const BG = "#0a0a0f";
const ACCENT_FROM = "#7c5cff";
const ACCENT_TO = "#2dd4bf";

/**
 * Build an SVG for the icon.
 * @param {number} px        canvas size
 * @param {number} glyphPct  glyph footprint as a fraction of the canvas (safe area)
 * @param {boolean} rounded  round the background corners (false for full-bleed maskable)
 */
function iconSvg(px, glyphPct, rounded) {
  const radius = rounded ? Math.round(px * 0.22) : 0;
  const fontSize = Math.round(px * glyphPct);
  // Nudge baseline so the cap-height "A" is optically centered.
  const baseline = Math.round(px / 2 + fontSize * 0.34);
  return `<svg xmlns="http://www.w3.org/2000/svg" width="${px}" height="${px}" viewBox="0 0 ${px} ${px}">
  <defs>
    <linearGradient id="g" x1="0" y1="0" x2="1" y2="1">
      <stop offset="0%" stop-color="${ACCENT_FROM}"/>
      <stop offset="100%" stop-color="${ACCENT_TO}"/>
    </linearGradient>
  </defs>
  <rect width="${px}" height="${px}" rx="${radius}" fill="${BG}"/>
  <text x="${px / 2}" y="${baseline}" font-family="system-ui, -apple-system, Segoe UI, Roboto, sans-serif"
        font-size="${fontSize}" font-weight="800" text-anchor="middle" fill="url(#g)">A</text>
</svg>`;
}

async function render(svg, outPath, px) {
  await sharp(Buffer.from(svg))
    .resize(px, px)
    .png()
    .toFile(outPath);
  console.log(`wrote ${outPath}`);
}

async function main() {
  await mkdir(OUT_DIR, { recursive: true });
  // "any" icons: glyph fills ~64% of the rounded square.
  await render(iconSvg(192, 0.64, true), path.join(OUT_DIR, "icon-192.png"), 192);
  await render(iconSvg(512, 0.64, true), path.join(OUT_DIR, "icon-512.png"), 512);
  // maskable: full-bleed field, glyph shrunk to ~46% so the circular mask safe
  // zone (inner ~80%) never clips it.
  await render(
    iconSvg(512, 0.46, false),
    path.join(OUT_DIR, "icon-512-maskable.png"),
    512,
  );
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
