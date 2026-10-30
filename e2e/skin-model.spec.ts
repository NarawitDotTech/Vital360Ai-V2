// Real-model check in a browser: onnxruntime-web (from jsDelivr, so this needs network) runs public/models/skin-v1.int8.onnx.
// 1) raw ORT-web logits on the Python-preprocessed tensor must match the Python int8 logits;
// 2) the full app flow (upload on /capture/skin) must report the real modelVersion and the same probabilities as Python.
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

test("app flow: real modelVersion and probabilities equal Python", async ({ page }) => {
  await page.goto("/th/consent/");
  await page.getByRole("checkbox").check();
  await page.getByRole("button", { name: "ยินยอมและเริ่มใช้งาน" }).click();
  await page.goto("/th/capture/skin/");
  await page.getByTestId("skin-file").setInputFiles(join(dir, "a.png"));
  await expect(page).toHaveURL(/\/th\/results\/$/, { timeout: 55_000 });
  await expect(page.getByTestId("result-skin")).toBeVisible();
  await expect(page.getByTestId("mock-notice")).toHaveCount(0);
  await page.getByRole("button", { name: /รายละเอียด/ }).click();
  await expect(page.getByText(skin.version)).toBeVisible();
  // probabilities shown (top classes, rounded to 1%) vs Python: same fixture, same temperature
  const p = softmax(expected.images[0].logits_int8, skin.temperature);
  const top = p.map((v, i) => [skin.labels[i], v] as const).sort((a, b) => b[1] - a[1])[0];
  const tierMax = Math.max(...["green", "amber", "red"].map((t) => p.reduce((a, v, i) => a + (skin.tiers[skin.labels[i]] === t ? v : 0), 0)));
  if (tierMax >= skin.threshold) await expect(page.getByTestId("similar-match")).toContainText(`${Math.round(top[1] * 100)}%`);
  else await expect(page.getByTestId("similar-match")).toHaveCount(0);
  const shown = await page.getByTestId("result-skin").innerText();
  expect(shown).toContain(`${Math.round(top[1] * 100)}%`);
});
