# Vitals 360 AI — Build Plan (planning only, build starts tomorrow)

## Context
Synapse Logic (CMU Demonstration School, M.2) needs a free, privacy-first health *screening* PWA: skin photo, cough/breath audio, and chest-video respiration, all inferred **on-device**, plus fusion modes and a dashboard that proves the project hypothesis (unimodal vs multimodal: Acc / Sens / Spec / F1). Source brief: `../prompt.md`. Project root: `Web-app/` (currently empty; `assets/icon.svg` not supplied yet → placeholder).
Tonight = plan only. No scaffolding, no installs, no agent files until approved.

## Key decisions (made per prompt §5 "don't ask for choices you can make")
| Area | Choice | Why |
|---|---|---|
| App | Next.js (App Router) `output: 'export'`, TypeScript, Tailwind | Matches original stack; static export = free hosting |
| Host | Cloudflare Pages (static) | Unlimited bandwidth matters for 5–15 MB model files; Vercel Hobby is fallback |
| Model hosting | Same-origin `/models/*.onnx` (≤15 MB each), cached by service worker | GitHub Release assets lack CORS; same-origin avoids it. HF Hub as fallback |
| Inference | `onnxruntime-web` (WebGPU → WASM) inside one Web Worker | Hard constraint #2; keeps main thread + JS budget small |
| Vision ROI | MediaPipe Tasks Vision (pose landmarker), lazy-loaded | Chest/shoulder ROI for respiration |
| Audio | Web Audio capture → log-mel in worker, TS port of training preprocessing | Must match Python exactly (parity test with fixtures) |
| i18n | Tiny in-house typed dictionary (`th.json`, `en.json`), `/th` `/en` routes, th default | No dependency needed for 2 locales |
| PWA | Hand-written `sw.js` (precache shell, runtime-cache models) | Fewer deps than Serwist/next-pwa |
| Backend | Supabase optional: auth, result summaries, consent log, opt-in feedback; RLS on all tables | Local-first; app fully usable with no account |
| Local store | IndexedDB (history, consent) | Offline + delete-my-data is trivial locally |
| Tests | Vitest (unit), Playwright (1 smoke E2E), Lighthouse CI budgets | Standard, few deps |
| Training | PyTorch on Kaggle (free T4/P100, weekly quota) primary, Colab free secondary | Small models fit a single free GPU |
| Skin model | MobileNetV3-Large / EfficientNet-Lite0, temp-scaled, Grad-CAM | Prompt §4 Phase 3 |
| Skin classes | 3-tier: benign / needs-review / urgent-referral (+ "uncertain" by threshold) | Few honest classes > 22 unreliable |
| Audio classes | normal / wheeze / crackle (ICBHI) + cough-abnormal head (Coswara/COUGHVID) | Prompt §4 Phase 2 example |
| Respiration | Classical: ROI motion/optical flow → bandpass 0.1–0.7 Hz → FFT peak → breaths/min | Learned model only if it beats this |
| Fusion | Late fusion: calibrated probs → logistic regression, weights shipped as JSON, run in TS | Tiny, explainable, no extra ONNX |

All versions get pinned exactly and checked online on day 1 (not from memory).

## Repo layout (to create tomorrow)
```
Web-app/
  .claude/agents/{frontend,backend,ai,reviewer}.md
  docs/{plan.md, data-report.md, gpu-training-playbook.md, agents/*.md, model-cards/*.md}
  app/            Next.js routes: [locale]/{consent,capture/{skin,audio,video},results,history,dashboard}
  src/inference/  index.ts (loadModel/predict/explain), worker.ts, mock.ts, mel.ts, resp.ts, fusion.ts
  src/i18n/  src/storage/  src/components/(Disclaimer, UrgentGuide, ...)
  public/{sw.js, manifest.json, models/, icons/}
  supabase/migrations/*.sql  supabase/functions/delete-account/
  training/{skin,audio,resp,fusion}/*.ipynb + common/ (splits, export, quantize, parity)
  eval/metrics/*.json     (written by training, read by dashboard)
  .env.example  README.md  .github/workflows/ci.yml
```
## Shared inference contract (the seam that unblocks everyone)
```ts
type Modality = 'skin' | 'audio' | 'resp' | 'fusion';
interface Prediction { probs: Record<string, number>; label: string | 'uncertain';
  confidence: number; extras?: { breathsPerMin?: number; heatmap?: ImageData }; modelVersion: string; latencyMs: number }
loadModel(m: Modality): Promise<void>       // fetch from manifest, SW-cached
predict(m: Modality, input): Promise<Prediction>
explain(m: Modality, input): Promise<Prediction['extras']>
```
`public/models/manifest.json` lists `{id, version, url, sha256, bytes, labels, thresholds, temperature}`. Mock backend returns deterministic fake probs until real ONNX lands; swapping is a manifest change.

## Sub-agents (written tomorrow as `.claude/agents/*.md`)
- **frontend**: app shell, i18n, PWA, consent, 3 capture flows with quality checks (light/noise/framing), results + disclaimers, history, dashboard. Uses only `src/inference`.
- **backend**: Supabase schema + RLS, optional auth, consent log, delete-my-data (local wipe + Edge Function deleting rows + auth user), CSP/HSTS headers via `_headers`, CI.
- **ai**: dataset research (web), `docs/data-report.md`, `docs/gpu-training-playbook.md`, `training/` notebooks, ONNX export + int8 + parity, model cards, `eval/metrics/*.json`.
- **reviewer** (read-only): gates each phase on correctness, security, privacy, medical wording, bundle size.
Each writes decisions to `docs/agents/<name>.md` (survives compaction).

## Phases & deliverables
**Day 1 – Phase 1 Foundation (frontend + backend in parallel, ai starts Phase 2 at the same time)**
- Scaffold, i18n th/en, consent screen (PDPA, explicit, logged locally), `<Disclaimer>` + `<UrgentGuide>` on every result, mock model, placeholder icon pipeline (generate favicon/180/192/512/maskable/OG from one SVG; sanitize SVG), SW + manifest, CI (lint, typecheck, vitest, build, Lighthouse budget: JS ≤200 KB gz).
- Supabase migration: `profiles`, `results` (summary only, no media), `consent_log`, `feedback` (opt-in); RLS `auth.uid() = user_id` on all.
- Exit: usable app with mock results, installable, offline after first load.

**Phase 2 Data discovery (ai, web research)** → `docs/data-report.md`
- Per dataset: URL, size, license, classes, biases, commercial/edu use. Check ISIC/HAM10000, PAD-UFES-20, Fitzpatrick17k, DDI, SCIN, DermNet (likely restricted — verify); ICBHI 2017, Coswara, COUGHVID, SPRSound; UBFC-rPPG, PURE, MMPD (several need signed agreements — say so). CXR marked optional/out of scope.
- Class mapping tables (original 22 skin classes → 3 tiers), fairness plan (per-Fitzpatrick metrics from Fitzpatrick17k/DDI/SCIN).

**Phase 3 Training (ai)** → `docs/gpu-training-playbook.md` + `training/`
- Verified current quotas/prices for Kaggle, Colab, Lightning, HF Spaces, RunPod, Vast, Lambda, Modal, student packs.
- Notebooks run top-to-bottom: download → patient-level split (fixed seed) → train (AMP, class weights/sampler, aug, checkpoint every epoch to `/kaggle/working` or Drive) → eval → temperature scaling → ONNX export → int8 (onnxruntime quantization) → parity check vs PyTorch → size report.
- One **shared test set** definition for fusion comparison (paired samples are rare across modalities → document how the fusion test set is built, e.g. synthetic pairing, and state this limitation honestly).

**Phase 4 Integrate & evaluate**
- Replace mock via manifest; benchmark latency on throttled mobile (Lighthouse/DevTools) and a real cheap Android if available.
- Dashboard: confusion matrices, per-class sens/spec, unimodal vs multimodal table, latency, size; CSV + PNG export. Reads `eval/metrics/*.json`.

**Phase 5 Extras (only after 1–4 pass)**: PM2.5 overlay (Air4Thai/open API), Thai FAQ-grounded tiny local LLM, clinic/รพ.สต. finder (OSM), longitudinal tracking + PDF, อสม. share, quizzes.

## Medical-safety & privacy rules baked in
- Never show a diagnosis; show tier + calibrated % + "uncertain" when max prob < threshold or capture quality fails.
- Th/En "screening only, see a doctor" + urgent-symptom list on every result.
- Media never leaves the device; backend stores summaries only; no fingerprinting analytics.
- `.env.example` only; Supabase anon key is public-by-design, service key never in the client.

## Verification (per phase, real output shown)
- `npm run lint && npm run typecheck && npm test && npm run build`
- Vitest units: i18n key parity th/en, consent gate, delete-my-data, mel parity vs Python fixture, respiration FFT on synthetic sine (e.g. 15 bpm → 15±1), fusion math.
- Playwright smoke: consent → capture (fake media) → result shows disclaimer.
- Lighthouse CI mobile: Perf ≥90, A11y ≥95, PWA installable; bundle budget check.
- Supabase: RLS test with two users (user B cannot read A).
- Training: parity max-abs-diff threshold, metrics only from held-out patient-level split.

## Tomorrow's first moves
1. `git init` in `Web-app/`, branch `feat/foundation`.
2. Verify current versions online, write `docs/plan.md` (this plan) + 4 agent files.
3. Launch frontend, backend, ai agents in parallel.

## Open items / risks
- No `assets/icon.svg` yet → placeholder icon + brand colors until the team supplies it.
- Paired multimodal data (same patient: skin + audio + video) likely doesn't exist publicly → fusion evaluation needs a clearly-stated method; possibly small team-collected pilot set (needs consent/ethics).
- Some datasets need signed DUAs (UBFC/PURE/MMPD, DermNet) → may not be usable in time.
- Camera respiration accuracy on cheap phones in low light is unproven → quality gate + "uncertain".
- Team PDFs in parent folder couldn't be text-extracted (no PDF tools installed); relied on prompt.md's summary.

## Elderly-friendly UX requirements (added 2026-10-27)
- Result screen leads with a plain-Thai one-sentence summary, a big traffic-light (green/amber/red) with icon + word, and one next step. Percentages and details sit below, collapsed.
- Never color alone: always icon + text. Base font ~18-20px, scalable, high contrast.
- One task per screen, big labeled buttons, a single "Start" path.
- Guided capture with on-screen and optional spoken prompts; "read aloud" for results (Web Speech API, Thai voice availability to verify).
- Gentle, jargon-free wording; urgent symptoms always clearly flagged.
- Plain failure messages ("too dark, move near a window"), never error codes.
- Verify with a small usability test with real older users once the app runs; Lighthouse cannot judge this.
