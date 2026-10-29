# Inference (src/inference, public/models)

UI imports only `@/inference` (`inference.loadModel / predict / explain`, plus `frameMotion`, `estimateBreathingRate`, `logMel`, types).

## Architecture
- Pure math (no DOM, unit-tested): `fft.ts`, `mel.ts`, `resp.ts`, `calibrate.ts`, `fusion.ts`, `mock.ts`.
- `engine.ts`: all predict/explain logic. `worker.ts` only forwards messages to it; `index.ts` talks to one Web Worker (`new Worker(new URL("./worker.ts", import.meta.url), {type:"module"})`, bundled by Turbopack; verified in a scratch static-export build). If `Worker` is missing or crashes, the same `Engine` runs on the main thread (tests, old browsers).
- Manifest: fetched from `/models/manifest.json` (no-cache); if that fails (offline first run, tests) the build-time copy is used.
- `predict` fills `latencyMs` and `modelVersion`. `explain` returns `undefined` (Grad-CAM is future work).
- `onnxruntime-web` is imported dynamically in `ort.ts` only when a skin/audio entry has a non-null `url` (WebGPU, then WASM). sha256 is verified when set (fail closed, also if `crypto.subtle` is missing). resp and fusion never use ONNX.

## Calibration
`probs = softmax(logits / temperature)`; if `max prob < threshold` then `label = tier = "uncertain"`; else tier from the manifest `tiers` map. Thresholds/temperatures in the manifest are placeholders until fitted on held-out data.

## Labels and tiers
| Modality | Label | Tier |
|---|---|---|
| skin | benign | green |
| skin | needs-review | amber |
| skin | urgent-referral | red |
| audio | normal | green |
| audio | wheeze, crackle, cough-abnormal | amber |
| resp | normal (12-20 /min) | green |
| resp | slow (<12) / fast (>20) | amber by label, escalated by rate (below) |
| fusion | low / moderate / high | green / amber / red |

Audio is never red: a short phone recording cannot establish urgency, so abnormal sounds only advise seeing a clinician.

Respiration tiers (adult resting, `respTier`): 12-20 green; 10-11 or 21-24 amber; <10 or >24 red. Label is only slow/normal/fast, so the engine overrides the label-default tier with the rate-based one.
Quality: SNR of the spectral peak mapped to 0..1; `< 0.3` => `uncertain` (breathsPerMin still returned in `extras`).

## Respiration pipeline
detrend (linear) -> Hann -> zero-padded FFT (x8) -> band 0.1-0.7 Hz (6-42 /min) -> peak + parabolic interpolation on log-magnitude. Needs >= 8 s of signal. `frameMotion(prev, cur, box)` is 1-D vertical optical flow (pixels, + = down) over subsampled luminance; the UI pushes one value per frame (or its running sum) as `signal`; the frequency is the same.

## Log-mel
16 kHz (box-average / linear resampler), n_fft 1024, hop 320, periodic Hann, centre reflect padding, 64 Slaney bands with Slaney normalisation, `log(1e-10 + power)`. Numeric parity with the Python training preprocessing is NOT verified: it needs a fixture (input wav + expected mel) exported from the training notebook (future work).

## Fusion
Features: per modality expected severity `P(amber)+2P(red)` (0..2). Uncertain/missing modality => `missingPrior` in `fusion-weights.json` (documented there). No usable part => `uncertain`. Weights are hand-set placeholders (`"untrained": true`).

## Mock backend
`mock.ts`: FNV-1a hash of simple input statistics (size, mean/std, zero-crossings...) seeds a PRNG that gives logits in [-2, 2]. Same input => same output, offline. It is not a classifier; every UI using it must say so.

## Swapping in a real ONNX model (manifest change only)
1. Put the file in `public/models/`, e.g. `skin-v1.onnx`.
2. Edit its entry in `public/models/manifest.json`: `url: "/models/skin-v1.onnx"`, `sha256` (hex), `bytes`, `version`, fitted `temperature` and `threshold`; labels/tiers must match the model output order.
3. Assumed I/O (override with `onnx: {input, output, imageSize, frames}`): skin `[1,3,224,224]` ImageNet-normalised; audio `[1,1,501,64]` log-mel; output `[1,nLabels]` logits. Different preprocessing means editing `ort.ts`.
4. ORT wasm files must be served same-origin (check CSP `wasm-unsafe-eval`, and that `public/_headers` allows the worker and wasm). The ORT path is untested against a real model.
