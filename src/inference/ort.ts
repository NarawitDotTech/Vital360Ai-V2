// onnxruntime-web backend. Imported dynamically, only when a manifest entry has a non-null url.
// UNVERIFIED against a real model (none exists yet). Assumed contract, overridable via entry.onnx:
//   skin : input [1,3,S,S] float32, RGB scaled 0..1 then ImageNet mean/std, S = onnx.imageSize ?? 224
//   audio: input [1,1,T,64] float32 log-mel (see mel.ts), T = onnx.frames ?? 501 (pad/crop), 16 kHz
//   output: [1,nLabels] raw logits.
import { logMel } from "./mel";
import type * as OrtApi from "onnxruntime-web";
import type { ModelEntry, ModalityInput } from "./types";

export interface LogitsRunner {
  run(input: ModalityInput, pixels: ImageData | null): Promise<number[]>;
}

async function sha256Hex(buf: ArrayBuffer): Promise<string> {
  const subtle = globalThis.crypto?.subtle;
  if (!subtle) throw new Error("crypto.subtle unavailable: cannot verify model hash");
  const d = new Uint8Array(await subtle.digest("SHA-256", buf));
  return Array.from(d, (b) => b.toString(16).padStart(2, "0")).join("");
}

// ORT's wasm binaries are 15-30 MB (over the 25 MiB per-file limit of the static host), so the runtime is NOT
// bundled: it is loaded at runtime from a pinned jsDelivr path, same version as package.json. Only the types
// come from the npm package (type-only import: nothing is emitted). Keep in sync with package.json and CSP.
const ORT_VERSION = "1.31.1";
const ORT_BASE = `https://cdn.jsdelivr.net/npm/onnxruntime-web@${ORT_VERSION}/dist/`;

async function loadOrt(): Promise<typeof OrtApi> {
  const ort = (await import(/* webpackIgnore: true */ /* turbopackIgnore: true */ `${ORT_BASE}ort.bundle.min.mjs`)) as typeof OrtApi;
  ort.env.wasm.wasmPaths = ORT_BASE;
  return ort;
}

export async function createOrtRunner(entry: ModelEntry): Promise<LogitsRunner> {
  if (!entry.url) throw new Error("createOrtRunner called without url");
  const res = await fetch(entry.url);
  if (!res.ok) throw new Error(`Model download failed (${res.status}): ${entry.url}`);
  const buf = await res.arrayBuffer();
  if (entry.sha256) {
    const got = await sha256Hex(buf);
    if (got !== entry.sha256.toLowerCase()) throw new Error(`Model ${entry.id}: sha256 mismatch`);
  }
  const ort = await loadOrt();
  let session;
  try {
    session = await ort.InferenceSession.create(buf, { executionProviders: ["webgpu", "wasm"] });
  } catch {
    session = await ort.InferenceSession.create(buf, { executionProviders: ["wasm"] });
  }
  const inName = entry.onnx?.input ?? session.inputNames[0];
  const outName = entry.onnx?.output ?? session.outputNames[0];

  return {
    async run(input, pixels) {
      let tensor;
      if (input.modality === "skin") {
        if (!pixels) throw new Error("skin: no pixel access in this context");
        tensor = new ort.Tensor("float32", skinTensor(pixels, entry.onnx?.imageSize ?? 224), [1, 3, entry.onnx?.imageSize ?? 224, entry.onnx?.imageSize ?? 224]);
      } else if (input.modality === "audio") {
        const T = entry.onnx?.frames ?? 501;
        const m = logMel(input.samples, input.sampleRate);
        const data = new Float32Array(T * m.nMels).fill(Math.log(1e-10));
        data.set(m.data.subarray(0, Math.min(m.data.length, data.length)));
        tensor = new ort.Tensor("float32", data, [1, 1, T, m.nMels]);
      } else {
        throw new Error(`ONNX not supported for ${input.modality}`);
      }
      const out = await session.run({ [inName]: tensor });
      return Array.from(out[outName].data as Float32Array);
    },
  };
}

function skinTensor(img: ImageData, S: number): Float32Array {
  const mean = [0.485, 0.456, 0.406];
  const std = [0.229, 0.224, 0.225];
  const out = new Float32Array(3 * S * S);
  for (let y = 0; y < S; y++) {
    const sy = Math.min(img.height - 1, Math.floor((y * img.height) / S));
    for (let x = 0; x < S; x++) {
      const sx = Math.min(img.width - 1, Math.floor((x * img.width) / S));
      const i = (sy * img.width + sx) * 4;
      for (let c = 0; c < 3; c++) out[c * S * S + y * S + x] = (img.data[i + c] / 255 - mean[c]) / std[c];
    }
  }
  return out;
}
