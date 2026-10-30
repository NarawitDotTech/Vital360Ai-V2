# GPU training playbook (Phase 3): student, no budget

Prices, quotas and rules checked **2026-10-30**. Tags: **[V]** read on the official page/API, **[S]** search snippet or third-party page, **[U]** unverified. These numbers change often: re-check the link before you rely on them.

**Important:** the notebooks in `training/kaggle/` were written and syntax-checked on a machine **without a GPU**. They have **not been run**. Expect to fix small things on the first run (that is what `SMOKE = True` is for). All runtimes below are estimates, not measurements.

## 1. Recommendation (read this first)

Our models are small (MobileNetV3-Large about 4 M params; audio CNN about 0.3 M). They train fine on **one free Kaggle T4 or P100**. Do not rent anything. Order of preference:

1. **Kaggle Notebooks** (free, weekly GPU quota, datasets attach in two clicks, output survives in "Output"). Use this for everything.
2. Google Colab free as a backup when your Kaggle quota is used up (not guaranteed).
3. Lightning AI free credits for a persistent Studio if you want a real terminal (limited, one-time).
4. Hugging Face Spaces only to show a demo, not to train.

Expected cost of the whole project: **0 THB**, about 1-2 GPU hours per training run (estimate), so 30 h/week is plenty for ~15 runs.

## 2. Free options

### 2.1 Kaggle (primary)
- **Quota**: weekly GPU quota is **30 hours "or sometimes higher depending on demand"**; it resets weekly on **Saturday 00:00 UTC** (Kaggle docs and a Kaggle staff post, read via search snippet [S]; the docs page itself did not render for my fetch tool, so check *Settings -> Your quota* on your own profile). Accelerators seen: T4 x2 and P100 [S]. TPUs are a separate pool [S]. Idle interactive sessions still burn quota [S]; interactive sessions end after 60 minutes idle [S]; a single session is historically up to 9 hours (some sources say 12) [S/U].
- **Phone verification** is required to turn on Internet and GPU [U] (it was in the past; confirm during setup).
- **Disk**: `/kaggle/working` keeps only about 20 GB and is what "Save Version" stores [U]. The notebooks keep caches under it; they are small.
- Official docs: https://www.kaggle.com/docs/notebooks and https://www.kaggle.com/docs/efficient-gpu-usage

**Exact steps (skin):**
1. Create a Kaggle account, verify your phone (Account -> Phone verification).
2. Go to https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000 , click **New Notebook** (this creates a notebook with the dataset already attached) .
3. In the new notebook: **File -> Import Notebook** and upload `training/kaggle/skin.ipynb` (or paste its cells). Delete the empty default cell.
4. Right sidebar **Session options**: Accelerator = **GPU T4 x2** (or **GPU P100**), Persistence = **Files only**, Internet = **On**.
5. **+ Add Input** (right sidebar) -> search `skin-cancer` -> add `mahdavi1202/skin-cancer` (PAD-UFES-20 phone photos; third-party upload, check it contains `metadata.csv` and `imgs_part_*`), or add the HF mirror data by `huggingface-cli download SalmaneExploring/pad-ufes-20 --repo-type dataset --local-dir /kaggle/working/pad` and edit the search path in cell "Find the data".
6. In config cell set `SMOKE = True`, click **Run All**. Everything should finish in a few minutes and print a metrics table. Fix any error, then set `SMOKE = False`.
7. **Save Version -> Save & Run All (Commit)** to run it in the background (you can close the tab). Output files appear under the notebook's **Output** tab: `skin_bundle.zip`. Download it.

**Audio**: same, with input `vbookshelf/respiratory-sound-database`; COUGHVID is downloaded inside the notebook (951 MB, needs Internet on).

**Expected runtime** (estimates, T4/P100): skin: image cache 5-10 min, training 15 epochs about 15-30 min, export + int8 + metrics about 5 min => **~30-50 min per run**. Audio: data load (ICBHI + COUGHVID decoding) 10-20 min, 40 epochs about 10-25 min, export 2 min => **~25-45 min**. GPU time is only the training part but Kaggle counts the whole session.

**If the session disconnects / times out / you hit the quota:**
- Training saves `ckpt_skin/last.pt` (or `ckpt_audio/last.pt`) **every epoch** (atomic write). Re-open the notebook, set the same accelerator, **Run All**: the loop prints `Resumed from epoch N`. With "Files only" persistence a re-opened interactive session usually keeps `/kaggle/working`; a new session does not. In that case: run once with *Save Version (Quick Save)* partway, then add the earlier version's output as an Input and copy `/kaggle/input/<your-notebook>/ckpt_skin/last.pt` to `/kaggle/working/ckpt_skin/last.pt` before Run All.
- Commit runs ("Save & Run All") are not killed by closing the browser; they stop only at the session limit or error.
- Out of quota: wait for Saturday 00:00 UTC (07:00 Thai time) or use Colab (2.2).
- Reproducibility: seed 42, group-level split saved to `skin_splits.csv` / `audio_splits.csv` inside the bundle, so the test set stays the same.

### 2.2 Google Colab (free)
- Free tier: notebooks run **at most 12 hours**, GPU types vary over time, resources "are not guaranteed and not unlimited", limits are not published, idle runtimes are deleted [V] (https://research.google.com/colaboratory/faq.html). Usually a T4 when available [U].
- Paid: Colab Pro **$9.99/month for 100 compute units**, pay-as-you-go **$9.99 per 100 units**, T4 burns about 1.76 units/hour [S] (Google's pricing page needs a login; not verified).
- The free Colab Pro for students offer (July 2025) is for **faculty and students at US-based higher-education institutions** only [V] (https://blog.google/products-and-platforms/products/education/colab-higher-education/), so **not available to a Thai M.2 student**.
- Steps: https://colab.research.google.com -> File -> Upload notebook -> `skin.ipynb` -> Runtime -> Change runtime type -> **T4 GPU** -> mount Drive for checkpoints (`from google.colab import drive; drive.mount('/content/drive')`) and change `WORK` in the config cell to `/content/drive/MyDrive/v360`. The notebooks look for data under `/kaggle/input`; on Colab download datasets yourself (HAM10000 from Dataverse, PAD-UFES-20 via the HF mirror) and point the glob to them. **Untested.**
- On disconnect: re-run all; `last.pt` on Drive resumes training.

### 2.3 Lightning AI
- Free plan per its pricing page [V] (https://lightning.ai/pricing): **up to 30 credits (5 at registration + 25 after adding a card), valid 12 months**, "roughly 80 GPU hours on interruptible machines"; one free Studio that restarts every 4 hours; free T4 hours are listed on the page (T4: 75, L4: 31) [V]. A phone number is required to prevent abuse [V]. A third-party tracker says the free tier became a one-time grant and T4 price rose to about $0.55/h in Sept 2026 [S]: treat the "75 h" figure as uncertain.
- Student/teacher programme: not found [V].
- Good for: a persistent Studio with VS Code in the browser; copy the repo, `pip install -r` the notebook's pip line, run the notebook with `jupyter` or convert to a script.

### 2.4 Hugging Face Spaces
- **ZeroGPU** is for *serving demos* (Gradio only), not training: free account **5 min/day** of GPU quota, PRO ($9/month [S]) **40 min/day**; free accounts may host 2 ZeroGPU Spaces [V] (https://huggingface.co/docs/hub/spaces-zerogpu). The backing GPU is a half RTX Pro 6000 Blackwell [V].
- Dedicated GPU Spaces cost per hour (T4 small about $0.40/h) [S].
- Use case here: a public ONNX demo or model card page. Model files can also be hosted on the Hub (https://huggingface.co/docs/hub) as a fallback to same-origin hosting.

## 3. Cheap options (only if a measured need appears)

| Service | Price (checked 2026-10-30) | Notes |
|---|---|---|
| Colab Pro | $9.99/mo, 100 units (about 55 h of T4) [S] | requires a card; not needed |
| Modal | Starter: **$30 free compute per month**, no base fee [V]. Per second: T4 $0.000164 (= $0.59/h), L4 $0.000222 ($0.80/h), A10 $0.000306 ($1.10/h), L40S $0.000542 ($1.95/h), A100 40GB $0.000583 ($2.10/h) [V] (https://modal.com/pricing) | Python SDK, needs code changes (not a notebook). Academic credits up to $10k are for graduate students/labs/researchers [V], not M.2 |
| RunPod | per hour, Community / Secure: RTX 4090 $0.34 / $0.89, RTX A5000 $0.16 / $0.27, L4 $0.44 / $0.59, A40 $0.35 / $0.59, A100 PCIe $1.19 / $1.79 [V, from the page's structured data] (https://www.runpod.io/pricing) | min deposit and payment methods not stated on page [U] |
| Vast.ai | marketplace, per-second billing; interruptible "50%+ cheaper" [V]; rough medians RTX 3090 about $0.18/h, RTX 4090 about $0.39/h [S, third-party, dates unclear] | prices vary daily; read https://vast.ai/pricing live; minimum deposit [U] |
| Lambda | on-demand: A10 24GB $1.29/h, A100 40GB $1.99/h, H100 PCIe $3.29/h [V] (https://lambda.ai/pricing) | needs card; overkill |

Rule of thumb: a 40-minute T4-class run costs about **$0.1-0.4** anywhere (e.g. RTX 4090 community $0.34/h). Payment by international credit/debit card is usually required; Thai prepaid cards may be refused [U]. Ask a parent/teacher to pay the first $5-10, only after Kaggle quota proved insufficient.

## 4. Student and education credits (reality check)
- **GitHub Student Developer Pack**: students aged **13+** with a school-issued email or dated proof of enrolment, any country [V/S] (https://education.github.com/pack ; GitHub blog "without a student ID"). Older GitHub help text lists secondary schools as eligible; I could not re-confirm on the current page [U]. The pack's partner offers change; do not assume it contains GPU credits.
- **Azure for Students**: $100 credit, 12 months, no card, school email [S]; one Microsoft Q&A answer says age 18+ [S/U], so an M.2 student is probably not eligible. Check https://azure.microsoft.com/free/students .
- **AWS Educate, Google Cloud/Google for Education credits**: not verified in this run [U].
- **Google Colab Pro for students**: US higher-ed only [V] (see 2.2).
- **Modal academic credits**: graduate students/labs only [V].
Conclusion: for this team, credits are not a realistic route; Kaggle free quota is.

## 5. Which hardware for which task
| Task | Fits on | Note |
|---|---|---|
| Skin MobileNetV3-L / EfficientNet-Lite0, 224 px, 12k images | 1x T4 or P100 16 GB, AMP, batch 64 | ~16 GB is more than needed |
| Audio small CNN, 501x64 input | any GPU, even CPU (slower) | decoding audio is the bottleneck |
| Distilled AST/PANNs | not recommended: size and a domain gap that bigger models will not fix | revisit only after the data issues in `docs/data-report.md` |
| Larger GPUs | not recommended without a measured need | |

## 6. The inference contract (what training MUST match)
Source of truth: `src/inference/ort.ts`, `src/inference/mel.ts`, `public/models/manifest.json` (read 2026-10-30).

**Common**: one ONNX file, same-origin URL, `sha256` and `bytes` in the manifest (the app checks sha256, fails closed). First input name is used unless `onnx.input` is set; first output unless `onnx.output`. Output = **raw logits** `[1, nLabels]`, class order = manifest `labels`. The app applies `softmax(logits / temperature)` and abstains if max prob < `threshold`. So: **no softmax and no temperature baked into the graph**. Our exporter names them `input` / `logits` and the manifest patch sets `onnx.input/output` explicitly. Model budget about 15 MB; ORT-web 1.31.1 loaded from jsDelivr, WebGPU then WASM.

**Skin**
- Input `float32 [1, 3, 224, 224]`, NCHW, RGB (alpha ignored).
- Normalisation: `x/255`, then `(x - mean) / std`, mean `[0.485, 0.456, 0.406]`, std `[0.229, 0.224, 0.225]` (ImageNet). Training does the same on the GPU.
- Resize: the **whole image is squashed** to 224x224 (aspect ratio not preserved, no centre crop), sampling with **nearest neighbour** (`floor(y*h/224)`). The notebook caches training images as 224x224 with PIL bilinear (antialiased) squash. **Mismatch**: for a 12-megapixel photo, nearest-neighbour sampling aliases. Recommended app-side fix (inference agent): pre-scale the photo to 224x224 with `drawImage` + `imageSmoothingQuality = "high"` before `getImageData`. Until then, expect lower accuracy on real phone shots than on the notebook's test split.
- Labels: `["benign", "needs-review", "urgent-referral"]`.

**Audio**
- Input `float32 [1, 1, 501, 64]` = `[batch, channel, frames, mels]`, raw log-mel with **no normalisation** in the app. The model therefore subtracts the per-clip mean and divides by a fixed training std **inside the ONNX graph** (gain-invariant).
- Features (`mel.ts`, `MEL_DEFAULTS`): 16 kHz (the app resamples with box average down / linear up), `n_fft` 1024, `hop` 320, periodic Hann window 1024, centre padding with **reflect**, power spectrum `|STFT|^2`, 64 mel bands 0-8000 Hz, **Slaney** scale and **Slaney area normalisation**, `log(1e-10 + mel_power)` (natural log), frames-major. Frames = `1 + floor(N/320)`: 160,000 samples (10 s) -> 501.
- Padding: if the clip gives fewer than 501 frames, the app fills the rest with `log(1e-10)` (= -23.03); if more, it crops. `common/mel.py` reproduces the front end with `torch.stft` + `librosa.filters.mel(htk=False, norm="slaney")`; the audio notebook writes `mel_fixture.json` (signal + expected log-mel). **Action for the inference agent**: add a vitest that feeds the fixture to `logMel` and compares with atol about 5e-3. Until that passes, parity between TS and Python is **not verified**.
- Resampling difference: the app box-averages 48 kHz -> 16 kHz; training loads with librosa (soxr). A small spectral difference above ~6 kHz is possible; training adds random band-limiting which makes the model tolerant.
- Labels: `["normal", "wheeze", "crackle", "cough-abnormal"]`.

## 7. After training: put the model into the app
1. Download `skin_bundle.zip` / `audio_bundle.zip`; unzip.
2. Copy `skin-v1.onnx` and/or `audio-v1.onnx` to `public/models/`. Each must be < 15 MB (the notebooks print the size and refuse to pick a variant that fails the gates).
3. Open `manifest_patch_*.json`; copy `version`, `url`, `sha256`, `bytes`, `threshold`, `temperature`, `onnx` into the matching entry of `public/models/manifest.json`. Keep `labels` and `tiers` as they are (the patch labels equal them).
4. Copy `metrics_skin.json` / `metrics_audio.json` into `eval/metrics/` (schema in `eval/metrics/README.md`; `isSample` is false). Delete or keep `sample.json` as you prefer. The fusion entry must be measured on the same test set (see `docs/data-report.md` section 7).
5. `latencyMs` in those files is ONNX Runtime **CPU on Kaggle**, not a phone. Measure on a real Android (Chrome DevTools remote debugging, median of 20 runs), then edit the number and say so in `testSetDescription`.
6. `npm run build && npm test`, then try a real photo/recording on a phone.
7. Do not commit files over 50 MB, datasets, or `ckpt_*/last.pt`.

## 8. What each notebook does (checklist)
`skin.ipynb`: install -> helper modules -> config + seed -> find data (HAM10000 required, PAD-UFES-20 optional) -> map to 3 tiers -> stratified group split (assert no group leaks) -> cache 224 px -> phone-like augmentation (crop, flips, rotation, colour jitter, white balance/exposure, blur, down-up scaling, JPEG, noise) -> MobileNetV3-L (or EfficientNet-Lite0) -> sqrt-inverse-frequency class weights + label smoothing -> AMP, checkpoint every epoch with resume, TensorBoard -> temperature scaling + abstain threshold on validation -> ONNX fp32 + parity (assert max |diff| < 1e-3) -> int8 QDQ and fp16 candidates, pick smallest passing gates (top-1 agreement >= 0.97, macro-F1 drop <= 0.02, size <= 15 MB) -> metrics JSON in dashboard schema + `report_skin.json` (ECE, abstain coverage, per source, per Fitzpatrick) -> manifest patch -> zip.
`audio.ipynb`: same shape; ICBHI cycles + COUGHVID expert-labelled coughs, 10 s canvases, band-limit/noise/gain/SpecAugment, small CNN with in-graph normalisation, fp32 shipped unless int8 is needed, plus `mel_fixture.json`.
Not implemented: Grad-CAM (the app's `explain` returns `undefined`), W&B (TensorBoard only), int8 quantisation-aware training, a real phone latency benchmark.

## 9. Troubleshooting
- `CUDA out of memory`: set `BATCH = 32`.
- `Internet` errors on pretrained weights: Session options -> Internet On (phone verification).
- ONNX export error mentioning `onnxscript`/dynamo: the helper calls the legacy exporter (`dynamo=False`); if your torch removed it, upgrade `onnxscript` and drop the kwarg in `common/export.py`.
- int8 loses accuracy: the notebook falls back to fp16/fp32 automatically; EfficientNet-Lite0 quantises more gracefully than hard-swish MobileNetV3 in many reports [U].
- COUGHVID column assertion fails: read the printed column list and edit `dcols`/`wcols` in the labelling cell.
- Nothing in the PAD-UFES-20 glob: check the dataset's CSV has `diagnostic` and `img_id` columns.
