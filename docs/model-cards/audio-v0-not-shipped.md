# Model card: audio-v0 (breath-sound CNN) - NOT SHIPPED, app audio stays MOCK

**Result: negative.** A small CNN trained on CPU on SPRSound did not beat the majority-class baseline in a useful way, so it is **not wired into the app** (`public/models/manifest.json` audio entry stays `mock-0.1.0`, no audio entry in `eval/metrics/`). Numbers are kept here so the attempt is reproducible and honest.

## Data
SPRSound (SJTU paediatric respiratory sound database, CC BY 4.0), via the Hugging Face mirror `Treza12/sprsound-classification` (1 949 recordings; the original GitHub release is larger). Paediatric (1 month to 18 years), electronic stethoscope, 8 kHz, ~9-15 s. The mirror's WAV headers are malformed (nBlockAlign 4 for mono 16-bit), read manually as 8 kHz mono PCM16 and resampled to 16 kHz.
Record-level labels mapped to the app's labels: Normal -> normal (1 303), CAS (continuous adventitious sounds: wheeze/rhonchi/stridor) -> wheeze (126), DAS (discontinuous: crackles) -> crackle (248). "CAS & DAS" and "Poor Quality" recordings dropped; 1 677 recordings kept. **cough-abnormal has no training data** (no cough set used) and is not in any model output. ICBHI 2017 was not reachable (HF mirrors have diagnosis labels only, no wheeze/crackle annotations).
Split: patient-level (`StratifiedGroupKFold(7)`, seed 42): train 1 196 / val 241 / test 240 recordings (35 test patients).

## Front end
`training/cpu/audio_mel.py` is a Python port of `src/inference/mel.ts` (16 kHz, n_fft 1024, hop 320, periodic Hann, centre reflect, 64 Slaney bands, `log(1e-10 + power)`), padded/cropped to 501 frames with `log(1e-10)` as `ort.ts` does. **Parity verified**: `src/inference/mel.test.ts` runs the TS `logMel` on a synthetic signal and a real SPRSound clip (`src/inference/__fixtures__/mel_fixture.json`, from Python) and matches within 5e-3.

## Attempts (all measured; the first three are on validation, only run 1 was also scored on test)
1. CNN (4 stride-2 conv blocks, 0.08 M params, in-graph normalisation, SpecAugment + gain), sqrt-balanced class weights, 30 epochs. Test (240 recordings): accuracy 0.775 (95 % CI 0.69 to 0.89), but the majority class alone gives 0.771. Sensitivity normal 0.96, wheeze 0.28 (n = 18), crackle 0.08 (n = 37); macro-F1 CI 0.32 to 0.59. Confusion (rows actual normal/wheeze/crackle): [[178,6,1],[12,5,1],[33,1,3]]. ONNX parity fp32 vs PyTorch max diff 1.5e-6.
2. Same CNN, fully balanced class weights: test accuracy 0.529, crackle sens 0.27 but specificity 0.63, normal sens 0.61. Worse overall.
3. Pooled log-mel statistics (per-band mean/std/p10/p90/flux) + logistic regression or gradient boosting: validation macro-F1 0.37 to 0.39, no better.

## Why it fails (diagnosis)
~1 200 training recordings with 126 wheeze examples, record-level (not event-level) labels, from 8 kHz paediatric stethoscopes, with a model trained from scratch. Adventitious sounds are short events inside a long recording, and a global label gives a weak signal. Next steps (GPU playbook): event-level crops from the JSON annotations, a pretrained audio backbone (PANNs / AST), more data (ICBHI, full SPRSound), and phone-microphone validation. Even a good stethoscope model would **not** transfer to phone recordings without testing on them.

## Files
`training/cpu/audio_mel.py`, `audio_prep.py`, `audio_train.py`, `audio_fixture.py`; measured numbers in `training/cpu/results/audio-v0-not-shipped.json`.
