import { defineConfig, devices } from "@playwright/test";

/**
 * Playwright E2E config. Builds the static site, serves the `out/` export, and
 * runs the browser specs in `tests-e2e/`. The web server is started by
 * Playwright and reused locally; in CI it boots fresh.
 */
export default defineConfig({
  testDir: "./tests-e2e",
  fullyParallel: true,
  forbidOnly: !!process.env.CI,
  retries: process.env.CI ? 2 : 0,
  reporter: process.env.CI ? "github" : "list",
  use: {
    baseURL: "http://localhost:4321",
    trace: "on-first-retry",
  },
  projects: [
    {
      name: "chromium",
      // CI uses Playwright's bundled Chromium (installed in the workflow). For
      // local runs where the bundled build can't be used, set PW_CHANNEL=chrome
      // (or "msedge") to drive a system browser instead.
      use: {
        ...devices["Desktop Chrome"],
        ...(process.env.PW_CHANNEL ? { channel: process.env.PW_CHANNEL } : {}),
      },
    },
  ],
  webServer: {
    // Build then serve the static export on a fixed port.
    command: "npm run build && npx sirv-cli out --single --port 4321 --quiet",
    url: "http://localhost:4321",
    reuseExistingServer: !process.env.CI,
    timeout: 180_000,
  },
});
