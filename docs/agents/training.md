# Training agent notes (CPU baseline)

Scope: real on-device models trained on this machine (4 CPU cores, 15 GB RAM, no GPU). Data, venv and caches live **outside the repo** in `/home/user/ml/` (`venv/`, `data/`, `cache/`, `runs/`, `export/`). Code: `training/cpu/`. Results kept in the repo: `training/cpu/results/`. Model cards: `docs/model-cards/`.

## What shipped / what did not
| Model | State | Notes |
|---|---|---|
| skin | **real**: `skin-v1.0.0`, 19 classes, 3.8 MB | `docs/model-cards/skin-v1.md` |
| audio | **mock** (negative result) | CNN did not beat majority baseline: `docs/model-cards/audio-v0-not-shipped.md`. `mel.ts` <-> Python parity IS verified (vitest) |
| resp | signal processing, no model | unchanged |
| fusion | untrained placeholder weights | no paired multimodal data exists; not faked |

## Decisions
- **Data:** HAM10000 (CC BY-NC 4.0, Harvard Dataverse), PAD-UFES-20 (CC BY 4.0, Mendeley public API), Google SCIN (HF `google/scin`, SCIN Data Use License) so that phone photos and darker skin tones are in the mix. SCIN cases kept only if the top dermatologist-weighted condition has weight >= 0.5 and beats the runner-up; classes with < 40 cases dropped (impetigo, keratosis pilaris).
- **Labels:** 19 fine classes; tier = argmax of summed class probabilities (`tierRule: "sum"` in the manifest, implemented in `src/inference/calibrate.ts`), uncertain when that tier probability < threshold. Types gained optional `ModelEntry.tierRule`, `Prediction.tierProbs` (backward compatible).
- **Split:** group-level (HAM lesion, PAD patient, SCIN case), seed 42, stratified 7-fold: fold 0 test, fold 1 val.
- **Model:** MobileNetV3-Large (ImageNet), blocks 0-6 frozen and cached, blocks 7-16 + head fine-tuned. A frozen-feature logistic-regression head was the first baseline (3-tier problem on HAM+PAD only, **validation** accuracy 0.76, macro-F1 0.68) and fine-tuning beat it clearly (validation accuracy 0.855), so fine-tuning was kept. (`skin_features.py`, `skin_train.py` are that baseline; `skin_train.py` is no longer on the shipped path.)
- **Preprocessing:** the app downscales photos to <= 640 px (`src/capture/camera.ts`); `skin-preprocess.ts` then does an **area-weighted** squash resize to 224x224 (the earlier nearest-neighbour sampling aliased on 12 MP photos), identical to `skin_common.py`. Parity test `src/inference/skin-preprocess.test.ts` (3 fixtures, tolerance 1e-3).
- **Quantisation:** static int8 QDQ of activations was tried and rejected (top-1 agreement with fp32 75.7 %; Percentile 85 %, Entropy 79 %). Shipped: per-channel int8 weights + DequantizeLinear (layers >= 20 000 weights; smaller layers fp32), activations fp32. 99.2 % top-1 agreement with fp32 on test, 3.8 MB. Speed is not improved over fp32 on desktop CPU (it is a size optimisation); a phone benchmark is still to do.
- **Not done / caveats:** Grad-CAM (`explain` still undefined), phone latency, a clinician review of tiers, fairness on dark skin (almost no data).

## Reproduce (about 2 h wall-clock on 4 cores, most of it downloads and CPU training)
```bash
python3 -m venv /home/user/ml/venv
/home/user/ml/venv/bin/pip install torch==2.9.1 torchvision==0.24.1 --index-url https://download.pytorch.org/whl/cpu
/home/user/ml/venv/bin/pip install -r training/cpu/requirements.txt pyarrow
cd training/cpu
# data -> /home/user/ml/data  (HAM10000: Dataverse files 3172585, 3172584, 4338392; PAD-UFES-20: Mendeley zr7vgbcyr2 imgs_part_1..3.zip + metadata.csv)
python skin_prep.py   /home/user/ml/data /home/user/ml/cache         # HAM+PAD -> 224px cache (~20 min)
python scin_prep.py   /home/user/ml/data/scin /home/user/ml/cache/scin 3   # SCIN shards, ~12 min, 3 workers
python skin_build_index.py /home/user/ml/cache                        # merge + label map -> index_all.csv, images_all.npy
python skin_finetune.py /home/user/ml/cache /home/user/ml/runs/ft19   # stage-1 cache ~12 min + 8 epochs x ~150 s; writes split, head.npz, backbone.pt
python skin_export.py /home/user/ml/cache /home/user/ml/runs/ft19 /home/user/ml/export   # ONNX + int8 + parity (~3 min)
python skin_eval.py   /home/user/ml/cache /home/user/ml/runs/ft19 /home/user/ml/export ../../eval/metrics/skin.json
python skin_fixture.py /home/user/ml/cache /home/user/ml/runs/ft19 /home/user/ml/export ../..   # parity fixtures
python skin_manifest.py /home/user/ml/runs/ft19 /home/user/ml/export ../..                      # copies the model, updates manifest.json
# audio (negative result): audio_prep.py, audio_train.py, audio_fixture.py (see the model card)
```
Then from the repo root: `npm run build && npx playwright test e2e/skin-model.spec.ts` (needs network: onnxruntime-web is loaded from jsDelivr).

## Timings (this machine)
HAM+PAD cache build ~20 min (one core); SCIN download+cache ~12 min; MobileNetV3 forward ~65 img/s (4 threads); fine-tune stage-1 cache ~12 min, 8 epochs ~21 min (~150 s per epoch on 10.9 k images); export+parity ~3 min; audio CNN 30 epochs ~2.5 min. Peak RAM ~6 GB.

## Gotchas
- `pip install` of torch from the PyTorch index hung once (killed and re-run, then fine).
- ORT static quantisation with Percentile/Entropy calibration holds all activations: use <= 64 calibration images or it is OOM-killed on 15 GB.
- The SPRSound mirror's WAV headers are invalid for scipy; `audio_prep.read_wav` parses them.
- `eval/metrics/skin.json` `models.skin` is the derived 3-tier view (the dashboard schema has one class list per model); the 19-class table, CIs and per-source/per-skin-tone breakdowns are in the extra keys of that file and in the model card.
