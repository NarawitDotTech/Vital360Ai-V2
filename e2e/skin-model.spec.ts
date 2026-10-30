// Real-model check in a browser: onnxruntime-web (from jsDelivr, so this needs network) runs public/models/skin-v1.int8.onnx.
// 1) raw ORT-web logits on the Python-preprocessed tensor must match the Python int8 logits;
// 2) the full app flow (upload on /capture/skin) must report the real modelVersion and the same probabilities / tier as Python;
// 3) every fixture (melanoma, BCC, eczema, healthy skin) gets the tier Python computes; 4) the camera guide box is square.
import { expect, test } from "@playwright/test";
import { readFileSync } from "node:fs";
import { join } from "node:path";

const dir = join(process.cwd(), "src", "inference", "__fixtures__", "skin");
const expected = JSON.parse(readFileSync(join(dir, "expected.json"), "utf8"));
const manifest = JSON.parse(readFileSync(join(process.cwd(), "public", "models", "manifest.json"), "utf8"));
const skin = manifest.models.find((m: { id: string }) => m.id === "skin");

const softmax = (l: number[], T: number) => {
  const m = Math.max(...l.map((x) => x / T));
  const e = l.map((x) => Math.exp(x / T - m));
  const s = e.reduce((a, b) => a + b, 0);
  return e.map((x) => x / s);
};

test("ORT-web reproduces Python int8 logits on the fixture tensors", async ({ page }) => {
  await page.goto("/th/");
  const result = await page.evaluate(
    async ({ u8, mean, std, url }) => {
      const base = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.31.1/dist/";
      const ort = await import(/* @vite-ignore */ `${base}ort.bundle.min.mjs`);
      ort.env.wasm.wasmPaths = base;
      const session = await ort.InferenceSession.create(await (await fetch(url)).arrayBuffer(), { executionProviders: ["wasm"] });
      const S = 224;
      const t = new Float32Array(3 * S * S);
      const bytes = Uint8Array.from(atob(u8), (c) => c.charCodeAt(0));
      for (let c = 0; c < 3; c++) for (let i = 0; i < S * S; i++) t[c * S * S + i] = (bytes[i * 3 + c] / 255 - mean[c]) / std[c];
      const out = await session.run({ input: new ort.Tensor("float32", t, [1, 3, S, S]) });
      return Array.from(out.logits.data as Float32Array);
    },
    { u8: readFileSync(join(dir, "a.u8.bin")).toString("base64"), mean: expected.mean, std: expected.std, url: skin.url },
  );
  const want: number[] = expected.images[0].logits_int8;
  expect(result.length).toBe(want.length);
  for (let i = 0; i < want.length; i++) expect(Math.abs(result[i] - want[i])).toBeLessThan(0.05);
});

async function consent(page: import("@playwright/test").Page) {
  await page.goto("/th/consent/");
  await page.getByRole("checkbox").check();
  await page.getByRole("button", { name: "ยินยอมและเริ่มใช้งาน" }).click();
}

test("app flow: real modelVersion, centre-square crop, probabilities / tier / label equal Python", async ({ page }) => {
  await consent(page);
  await page.goto("/th/capture/skin/");
  await page.getByTestId("skin-file").setInputFiles(join(dir, "a.png")); // 4:3 PNG: the app must crop the centre square
  await expect(page).toHaveURL(/\/th\/results\/$/, { timeout: 55_000 });
  await expect(page.getByTestId("result-skin")).toBeVisible();
  await expect(page.getByTestId("mock-notice")).toHaveCount(0);
  await page.getByRole("button", { name: /รายละเอียด/ }).click();
  await expect(page.getByText(skin.version)).toBeVisible();
  const im = expected.images[0];
  await expect(page.getByTestId("result-skin")).toHaveAttribute("data-tier", im.tier);
  const p = softmax(im.logits_int8, skin.temperature);
  const pct = Math.round(p[skin.labels.indexOf(im.label)] * 100);
  if (im.tier === "uncertain") await expect(page.getByTestId("similar-match")).toHaveCount(0);
  else await expect(page.getByTestId("similar-match")).toContainText(`${pct}%`);
});

test("all fixtures (incl. a healthy-skin patch) give the Python tier through the real app", async ({ page }) => {
  test.setTimeout(240_000);
  await consent(page);
  for (const im of expected.images as { name: string; tier: string; label: string }[]) {
    await page.goto("/th/capture/skin/");
    await page.getByTestId("skin-file").setInputFiles(join(dir, `${im.name}.png`));
    await expect(page).toHaveURL(/\/th\/results\/$/, { timeout: 55_000 });
    await expect(page.getByTestId("result-skin"), im.name).toHaveAttribute("data-tier", im.tier);
  }
});

test("camera: the guide box is square and the photo is analysed", async ({ page }) => {
  await consent(page);
  await page.goto("/th/capture/skin/");
  await page.getByRole("button", { name: "เปิดกล้อง" }).click();
  const box = page.locator(".guide-box");
  await expect(box).toBeVisible();
  const b = (await box.boundingBox())!;
  expect(Math.abs(b.width - b.height)).toBeLessThan(2);
  await page.getByRole("button", { name: "ถ่ายรูป" }).click();
  await expect(page).toHaveURL(/\/th\/results\/$/, { timeout: 55_000 });
  await expect(page.getByTestId("result-skin")).toBeVisible();
});

test("a skin photo with a big black spot is not shown green", async ({ page }) => {
  // synthetic stress test: a PAD-UFES-20 skin patch (CC BY 4.0) with a black ellipse painted on (e2e/fixtures/skin-blackspot.jpg)
  await consent(page);
  await page.goto("/th/capture/skin/");
  await page.getByTestId("skin-file").setInputFiles(join(process.cwd(), "e2e", "fixtures", "skin-blackspot.jpg"));
  await expect(page).toHaveURL(/\/th\/results\/$/, { timeout: 55_000 });
  const tier = await page.getByTestId("result-skin").getAttribute("data-tier");
  expect(["amber", "red"]).toContain(tier);
});
