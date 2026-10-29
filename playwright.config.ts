import { defineConfig, devices } from "@playwright/test";
import { existsSync, readdirSync } from "node:fs";
import { join } from "node:path";

const PORT = 4173;

// The pre-installed Chromium can be a different build than the one this Playwright version expects.
// Use CHROMIUM_PATH if set, else any chromium-*/chrome-linux64/chrome under PLAYWRIGHT_BROWSERS_PATH, else Playwright's own.
function chromiumPath(): string | undefined {
  if (process.env.CHROMIUM_PATH) return process.env.CHROMIUM_PATH;
  const base = process.env.PLAYWRIGHT_BROWSERS_PATH;
  if (!base || !existsSync(base)) return undefined;
  const dir = readdirSync(base).find((d) => /^chromium-\d+$/.test(d));
  const exe = dir && join(base, dir, "chrome-linux64", "chrome");
  return exe && existsSync(exe) ? exe : undefined;
}

// Serves the static export in out/ (run `npm run build` first).
export default defineConfig({
  testDir: "./e2e",
  timeout: 60_000,
  expect: { timeout: 10_000 },
  fullyParallel: false,
  retries: 0,
  reporter: [["list"]],
  use: {
    baseURL: `http://localhost:${PORT}`,
    trace: "retain-on-failure",
    serviceWorkers: "allow",
  },
  webServer: {
    command: `node scripts/serve-out.mjs ${PORT}`,
    url: `http://localhost:${PORT}/th/`,
    reuseExistingServer: true,
    timeout: 30_000,
  },
  projects: [
    {
      name: "chromium",
      use: {
        ...devices["Pixel 7"],
        permissions: ["camera", "microphone"],
        launchOptions: {
          executablePath: chromiumPath(),
          args: ["--use-fake-ui-for-media-stream", "--use-fake-device-for-media-stream"],
        },
      },
    },
  ],
});
