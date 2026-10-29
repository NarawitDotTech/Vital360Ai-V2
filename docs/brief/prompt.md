# Vitals 360 AI: Rebuild Prompt for Claude Code

You are the **lead orchestrator** for rebuilding **Vitals 360 AI** from scratch, as a fast, free-to-host, privacy-first health screening web app with **real, trained AI models** that run **on the user's own device**.

You have internet access. Use it to find real datasets, current library versions, and current free GPU options. Do not rely on memory for anything that changes (versions, quotas, dataset licenses). Verify first.

## 0. Project background (from the team's own documents)

- Team: **Synapse Logic** (Chiang Mai University Demonstration School, M.2 Data Science class). Advisor: Mr. Apirat Angkabut.
- Official title: *Development of an All-in-One AI System for Dermatological & Respiratory Screening*.
- Problem: COPD is the #3 cause of death worldwide (about 3.3M per year). Lung cancer is #2 among cancer deaths in Thailand. PM2.5 is driving skin disease. Specialists in remote areas can be as few as 1 per 50,000 people. Diagnostics cost 3,000 to 10,000 THB.
- Goal: free early screening on a phone, with no special hardware, using only camera and microphone.
- Original stack: Next.js + TypeScript (TSX), hosted on Vercel, Supabase for the database, PyTorch for training, models run **locally on the user's device** (PDPA privacy).
- Original test window: 11 Jul to 15 Aug 2569 BE. Budget so far is about 1,470 THB total (cloud DB, Claude Pro, ChatGPT Pro). **Assume a near-zero budget.**

### The five test modes the project must support (from the proposal's results table)
1. **Unimodal image**: skin lesion photo classification.
2. **Unimodal audio**: cough and breathing sound (wheeze, crackle, normal) classification.
3. **Unimodal video / computer vision**: respiratory rate and chest movement from the phone camera (no contact).
4. **Multimodal**: lesion image + vital-sign / sound signals fused.
5. **All-in-one multimodal**: image + video + audio fused into one risk report.

The app must let the team **compare unimodal vs multimodal accuracy** (Accuracy, Sensitivity, Specificity, F1) because that is the project's hypothesis. Build an evaluation dashboard for it.

### Skin classes in the original plan
acne, actinic keratosis, benign tumors, bullous disease, candidiasis, drug eruption, eczema, infestations/bites, lichen planus-type disorders, lupus/connective tissue disease, moles/nevi, psoriasis, rosacea, seborrheic keratosis, skin cancer, sun damage, tinea/fungal, vascular tumors, vasculitis, warts, normal, unknown. Start by collapsing these into a **smaller, defensible class set** (see Phase 2) before training.

## 1. Hard constraints (non-negotiable)

1. **Lightweight and free to host.** Target: the whole app runs on Vercel/Cloudflare free tier plus Supabase free tier. **No GPU server, no always-on backend for inference.**
2. **Inference runs in the browser** (ONNX Runtime Web with WebGPU/WASM fallback, or TF.js). Images, audio and video **never leave the device** by default.
3. **Works on a cheap Android phone** over a slow connection. Initial JS under about 200 KB gzipped, each model under about 15 MB (quantized int8/fp16), models lazy-loaded and cached by a service worker (PWA, offline capable).
4. **Not a medical device.** Every result screen shows a clear "screening only, not a diagnosis, see a doctor" notice in Thai and English, with urgent-symptom guidance. Never output a definitive diagnosis. Report calibrated probabilities and an "uncertain, please consult a clinician" state.
5. **PDPA-aware.** Explicit consent screen, no analytics that fingerprint users, optional account only to sync history, easy delete-my-data.
6. **Thai first, English second** (i18n from day one). Accessible (WCAG 2.2 AA target, keyboard, screen reader labels, high contrast, large touch targets). Note: full accessibility validation still needs manual testing with assistive tech.
7. **Secrets** never committed. Use env vars and a `.env.example`.

## 2. Architecture (keep it tiny)

```
Phone / Browser (PWA)
 ├─ Next.js static export (or Vite + React), TypeScript, Tailwind
 ├─ ONNX Runtime Web (WebGPU -> WASM fallback), models cached in IndexedDB/Cache API
 ├─ MediaPipe (face/pose/ROI) for chest and nose region tracking
 ├─ Web Audio API for mel-spectrograms, Web Worker for all inference
 └─ Optional: Supabase (auth + history + anonymized opt-in feedback), RLS on every table

Training (offline, free GPUs, run by the AI agent's instructions)
 └─ PyTorch -> export ONNX -> int8 quantize -> publish to repo release / Hugging Face Hub
```

Rules:
- Prefer **static hosting**. Add a serverless function only if there is no alternative.
- No model is ever served from the user's one free server unless it is a plain static file host. The "free server" is for the web app and optional database only. It is **not** used for training or inference.
- Keep dependencies few and pinned to exact versions. Flag any package whose name looks unusual.

## 3. Sub-agents (create these as `.claude/agents/*.md`, then delegate)

Create three project sub-agents and one reviewer. Delegate independent work to them in parallel. Each agent writes its findings and decisions to `docs/agents/<name>.md` so context survives compaction.

### `frontend` (UI/UX, Thai-first, mobile-first)
- Owns: Next.js app, PWA/service worker, i18n (th/en), capture flows (camera photo, 15 to 30 s chest video, 10 s cough/breath audio), consent screen, results screen, history, evaluation dashboard.
- Guides the user with on-screen framing guides, lighting and noise checks before recording, and clear plain-language results.
- Runs inference only through the shared `inference` module in a Web Worker.
- Verifies with Lighthouse (performance, accessibility, PWA) and tests on a throttled mobile profile.

### `backend` (thin, free-tier, secure)
- Owns: Supabase schema + Row Level Security, optional auth, history sync, consent log, delete-my-data endpoint, model manifest/versioning, CI/CD, deployment config.
- Defaults to **local-first**: the app must be fully usable with no backend. Backend is an enhancement.
- Never stores raw images, audio, or video unless the user explicitly opts in. Store only result summaries.
- Adds rate limits, input validation, and security headers (CSP, HSTS). Flags any endpoint without authentication.

### `ai` (research, data, training, export)
- Owns: dataset discovery, licensing review, preprocessing, training scripts, evaluation, calibration, ONNX export and quantization, model cards.
- Searches the internet for datasets and papers, then **writes a ranked dataset report** (see Phase 2).
- Produces a **GPU training playbook** (see Phase 3) telling the user exactly how to train on powerful free or cheap GPUs, step by step, with copy-paste notebooks.
- Never claims accuracy it did not measure. Reports metrics on a held-out, patient-level split.

### `reviewer` (read-only)
- Reviews each phase for correctness, security, privacy, medical-safety wording, and bundle size before it is accepted.

## 4. Phases (do them in order, ship something usable at the end of each)

### Phase 1: Foundation
- Scaffold the app, i18n, PWA, consent flow, disclaimer components, CI, Lighthouse budget checks.
- Define the shared `inference` interface (`loadModel`, `predict`, `explain`) so models are swappable.
- Ship a working UI with a **mock model** first, so the frontend is never blocked on training.

### Phase 2: Data discovery (ai agent, uses the internet)
Search for and evaluate open datasets. For each, record: name, URL, size, license, classes, known biases (skin tone, device, country), and whether commercial or educational use is allowed. Candidates to check first (verify current availability and license, do not assume):
- **Skin**: ISIC Archive, HAM10000, DermNet, Fitzpatrick17k, PAD-UFES-20, DDI (diverse skin tones), SCIN (Google).
- **Respiratory audio**: ICBHI 2017 Respiratory Sound Database, Coswara, COUGHVID, Virufy, SPRSound.
- **Chest X-ray (optional, advanced)**: NIH ChestX-ray14, CheXpert, MIMIC-CXR (requires credentialed access, say so honestly).
- **Camera vital signs (rPPG / respiration)**: UBFC-rPPG, PURE, MMPD, and any chest-motion video sets.

Deliver `docs/data-report.md` with a recommended minimal class set, for example: *benign / needs-review / urgent-referral* for skin, and *normal / wheeze / crackle / cough-abnormal* for audio. Explain the trade-off between many classes (impressive but unreliable) and few classes (honest and reliable). **Prefer few classes with good metrics.**

Address fairness: report accuracy per skin tone (Fitzpatrick) where labels exist, and state limitations plainly.

### Phase 3: Real training (ai agent writes a GPU playbook)
Produce `docs/gpu-training-playbook.md` for a student with no GPU and no budget. Cover, with current quotas verified online:
1. **Free options**: Kaggle Notebooks (weekly free GPU hours), Google Colab free tier, Lightning AI free studio hours, Hugging Face Spaces for demos.
2. **Cheap options**: Colab Pro, RunPod, Vast.ai, Lambda, Modal credits, and any student or education credits (GitHub Student Pack, Google for Education, AWS Educate, Azure for Students).
3. A recommendation by task: small models (MobileNetV3, EfficientNet-Lite, a small audio CNN / AST-tiny) train fine on a **single free T4 or P100**. Do not recommend renting large GPUs unless a measured need exists.
4. Ready-to-run notebooks/scripts in `training/`: data download, preprocessing, train, evaluate, calibrate, export. Each cell must run top to bottom. Include seeds, patient-level splits, class-imbalance handling, augmentation, mixed precision, checkpointing to survive session timeouts, and Weights & Biases or TensorBoard logging.
5. **Export path**: PyTorch -> ONNX -> int8 quantization -> verify parity against the PyTorch outputs -> record final size and latency on a mid-range phone.
6. Exact commands the user pastes, expected runtimes, expected cost, and what to do when the free session disconnects.

Models to build (small first, scale only if metrics justify it):
- **Skin**: fine-tune MobileNetV3 or EfficientNet-Lite (ImageNet-pretrained). Add temperature-scaling calibration and Grad-CAM for explanations.
- **Audio**: log-mel spectrogram -> small CNN or a distilled AST/PANNs-tiny, trained on cough and breathing sounds.
- **Camera respiration**: classical signal processing first (optical flow / ROI intensity -> bandpass -> FFT peak) with MediaPipe landmarks. Only add a learned model if it beats the baseline. Validate against a reference signal where possible.
- **Fusion**: late fusion first (calibrated probabilities -> small logistic regression or MLP), then try learned fusion. Compare against each unimodal model on the **same test set**, as the proposal requires.

### Phase 4: Integrate and evaluate
- Load the exported ONNX models in the app, replace the mock model, benchmark on a throttled mobile profile.
- Build the **evaluation dashboard**: confusion matrices, per-class sensitivity and specificity, unimodal vs multimodal table, latency, model size. Export a CSV and a chart so the team can paste results into its report.

### Phase 5: Cool extras (only after Phases 1 to 4 pass)
- PM2.5 / air-quality overlay using a free open API, with a risk note for skin and lung symptoms.
- On-device **symptom chat assistant** using a tiny local LLM (WebLLM or Transformers.js) for plain-language explanations in Thai, grounded in a vetted FAQ, never giving diagnoses.
- Nearest clinic / รพ.สต. finder using open map data.
- Longitudinal tracking (lesion photo comparison over time, breathing-rate trend charts), exportable PDF for a clinician.
- Offline mode and a one-tap "share with village health volunteer (อสม.)" flow.
- Lightweight gamified health literacy quizzes in Thai.

## 5. Working rules for you (the orchestrator)

- Start by reading this file fully, then write a short plan to `docs/plan.md` and begin. Do not stop to ask for approval of choices you can make yourself. List your choices in your final message.
- Use sub-agents in parallel for independent work (frontend + backend + ai at the same time). Keep your own context for integration.
- Before declaring any phase done: run the build, lint, type-check, and tests, and show the real output. If something fails, say so with the output.
- Write tests for new logic (signal processing, fusion, i18n, consent, data deletion).
- Commit in small steps on a feature branch. Never push to main. Do not commit datasets, model weights over 50 MB, or secrets (use Git LFS or release assets).
- Be honest about limits: report only metrics you measured, list datasets you could not access, and mark anything unverified.
- If an approach fails twice, stop patching, diagnose the root cause, and try a different approach.

## 6. Definition of done

- [ ] App deploys on free tier, installs as a PWA, and works offline after first load.
- [ ] All five test modes work on-device, with real trained models (not mocks).
- [ ] Thai and English UI, consent flow, disclaimers, delete-my-data.
- [ ] Lighthouse mobile: Performance >= 90, Accessibility >= 95, PWA passing.
- [ ] `docs/data-report.md`, `docs/gpu-training-playbook.md`, model cards, and the evaluation dashboard exist.
- [ ] Unimodal vs multimodal comparison measured on one shared test set.
- [ ] README explains setup in under 10 commands.

## 6b. Brand icon

The team supplies the app icon as an SVG at `assets/icon.svg`. If it is missing, use a plain placeholder and note it in your final message. From that single file, generate the favicon, `apple-touch-icon` (180 px), PWA icons (192 and 512 px, plus a maskable 512 px with safe-zone padding), and the Open Graph image. Inline or reuse the SVG in the header logo. Sanitize it (strip scripts and external references) before bundling. Take the brand colors from the icon for the theme and `manifest.json` `theme_color`, and check text contrast against them.

## 7. First message to send yourself

"Read prompt.md. Create the three sub-agents. In parallel: frontend scaffolds the app with a mock model, backend defines the Supabase schema and RLS, ai searches the internet for datasets and writes docs/data-report.md. Then continue through the phases."

