# Model card: skin-v1.1.0 (20-class skin image screening baseline, "skin-v2" pipeline)

**Status:** real, trained baseline for a school project. **Not a medical device, not a diagnosis.** All numbers are measured on a held-out test split (never used for training, epoch selection, calibration or threshold choice). Tier assignments of the 11 SCIN-derived classes are provisional and **need clinician/teacher review**. The previous release (skin-v1.0.0) is documented unchanged further below; its numbers are the "before" column here.

## What changed in skin-v1.1.0 and why
Field problem: normal skin photographed with a phone was shown as a disease, and a photo with a big black spot as green "no concern". Diagnosis of the live app (browser = onnxruntime-web, real ONNX, real PAD-UFES-20 test photos at full resolution, upload path):
- Preprocessing matched Python (EXIF applied by `createImageBitmap`, RGBA read as RGB, no mirroring of pixels, normalisation identical); the only deviation was the canvas down-scale (max class-probability difference vs Python: mean 0.03, max 0.10 over 12 photos; one top-1 flip on a 0.38/0.38 tie). RGBA with transparent pixels would be composited on black (fixed: white).
- The real causes were (1) the camera path sent the **whole 4:3 frame** (the guide box was only drawn, never cropped), the upload path the whole photo, squashed to a square: the model saw "an arm", not "a spot"; (2) the model **had no "no lesion" class**: on 574 healthy-skin test patches skin-v1.0.0 answered actinic keratosis 54 %, dermatofibroma 24 %, nevus 7 %, tinea 6 % (and showed amber/red for 30 %); (3) tier = argmax of tier sums needs a lot of evidence for amber/red: only 60 % of urgent cases were shown red on this test set (73 % on phone photos).

Changes (all measured below): square crop (camera: exactly the on-screen guide box, 70 % of the frame height; upload: centred square) in app, training and evaluation; new class **healthy-skin** (tier green; i18n: th "ไม่พบรอยโรคที่ชัดเจน", en "No visible skin condition"); sensitivity-first tier rule `tierRule: "thresholds"`; fine-tune from skin-v1.0.0 with phone-like augmentation and PAD-UFES-20 / SCIN up-sampling; temperature re-fitted; capture quality check relaxed for smooth skin; guide text "fill the box with the spot".

## Input pipeline (app = training = evaluation)
decode (EXIF applied) -> square crop: camera = the square under the guide box (`guideRect`, 0.70 x visible frame height, centred), upload = centred square of the short side (`centerSquare`) -> if larger than 448 px, shrink to 448 px (canvas, repeated <= 2x draws, so close to a box average; Python: PIL BOX) -> area-weighted resize to 224 x 224 -> /255 -> ImageNet mean/std -> NCHW float32. Python: `training/cpu/skin_pipe.py`; TS: `src/capture/camera.ts`, `src/inference/skin-preprocess.ts`. After the fix, browser vs Python probabilities on 15 full-resolution photos: mean max difference 0.011, max 0.036, top-1 agrees on 14/15.

## Decision rule (`tierRule: "thresholds"`, `src/inference/calibrate.ts`, mirrored in `training/cpu/skin_decide.py`)
`probs = softmax(logits / T)`, T = 0.8682; tier probabilities = sums of class probabilities.
1. **red** if P(red) >= 0.185;
2. else **amber** if P(amber) + P(red) >= 0.335;
3. else **amber** if the top class is nevus or seborrheic keratosis, melanoma is the runner-up and P(melanoma) >= 0.05 (the "melanoma hiding behind a look-alike" guard);
4. else **amber** if an enclosed very dark spot covers >= 3 % of the box (`darkSpot`: 56 x 56 luma grid, dark = luma < 0.45 x median and < 70, 4-connected blob that does not touch the border; this is what stops a black spot from being green);
5. else **uncertain** if P(green) < 0.66 (the skin-v1 rule, kept; inert in practice because a green answer already has P(green) >= 1 - 0.335);
6. else **green**.
Shown label = most likely class of the shown tier (guard / dark-spot: the overall top class). All cut-offs were chosen on the VALIDATION split only: t_red = largest value that still shows >= 0.90 of validation urgent cases as red (val: 0.903 at 0.185); t_amber = largest value that still shows >= 0.80 of amber-tier cases as amber or red; guard floor = largest P(melanoma) floor that keeps the guard's full validation gain on cancers (the literal rule with floor 0 raises 37 % of validation nevi to amber for the same gain on cancers, a floor of 0.05 only 15 %; set `guard.minWatchProb: 0` for the literal rule). The dark-spot area (3 %) was fixed a priori, not tuned (it flags 0 of 574 healthy patches and 1.4 % of PAD+SCIN test images, but 20 % of dermoscopy images, which are not phone photos).

## Model
- Backbone: torchvision MobileNetV3-Large; blocks 0-6 frozen (ImageNet), blocks 7-16 and the head (960 -> 20) fine-tuned from the skin-v1.0.0 weights (new head row zero-initialised). AdamW, one-cycle LR (1.5e-4 / 1.5e-3), 8 epochs x 14 000 draws, label smoothing 0.05, dropout 0.3, sqrt-balanced class weights on the drawn distribution, BatchNorm frozen. Best epoch (7) by validation macro-F1.
- Sampling per epoch (relative weights per image): PAD-UFES-20 lesions 3.0, SCIN lesions 2.0, HAM10000 0.8 (nevus 0.3), PAD healthy patches 1.5, SCIN healthy patches 1.0.
- Augmentation (online, on 256 px square crops): random-resized crop (45-100 % area, aspect 0.8-1.25), flips / 90 deg turns / small rotation, Gaussian blur (35 %), per-channel white-balance gains (0.88-1.12), brightness, contrast, saturation, gamma, uneven-lighting gradient + vignette (45 %), noise (30 %), JPEG re-compression q35-92 (50 %).
- File: `public/models/skin-v1.1.int8.onnx`, **3.82 MB** (limit 15 MB); same weight-only per-channel int8 recipe as v1.0.0. fp32 vs int8 on the test images: max |logit diff| 0.227, mean 0.025, top-1 agreement 0.9942. Desktop CPU 6.1 ms/image (not measured on a phone).

## Data (all public, legally usable for education/research; raw data is not in the repo)
Same three sources and licences as skin-v1.0.0 (table below) plus **healthy-skin patches** cut from the same photos: PAD-UFES-20 (square crops of 28-45 % of the short side, outside a central disc of radius 0.2 x short side where the lesion is, <= 2 per photo) and SCIN (<= 1 per photo, any shot type). A patch is kept only if it passes automatic checks (`skin_pipe.py patch_ok`): >= 93 % skin-coloured pixels (YCbCr), no dark blob (min blurred luma >= 0.8 x median), no bright blob, no red/pale blob, no small red spots, hue ratio in the skin range, low contrast, almost no dark pixels (hair). Result: 4 192 patches (PAD 1 357, SCIN 2 835; the cap of 3 200 for SCIN was not reached). **Spot check:** `docs/model-cards/skin-v1.1-healthy-patches.jpg` (128 random patches, rows: PAD train, SCIN train, PAD test, SCIN test) was inspected: almost all are plain skin of many tones with hair, pores and freckle-level texture; roughly 3-5 % are questionable (faint redness or tiny dots, a wood-grain-like patch). These cannot be removed without a clinician, so some healthy "ground truth" is noisy, which makes the healthy false-alarm numbers slightly pessimistic. No further clearly licensed healthy-skin source was found and verified, so none was added. SCIN has no "normal" label, so SCIN patches are skin regions of people who photographed a skin concern: "healthy" means "no visible lesion in this patch", not "this person has no skin condition".

### Split (leak-free)
skin-v1.0.0 split kept as is for all lesion images (group-level: HAM lesion_id, PAD patient_id, SCIN case_id; seed 42; fold 0 test, fold 1 val). A healthy patch inherits the split of its source group (PAD patient / SCIN case); SCIN cases that were not in the v1 split (no class label) are assigned by a hash of the case id (1/7 test, 1/7 val). Asserted: no group is in two splits. Sizes: train 13 924, val 2 723, test 2 746 (test: HAM10000 1 423, PAD-UFES-20 517 of which 181 healthy, SCIN 806 of which 393 healthy). Table: `training/cpu/results/skin-v1.1-split.csv.gz`.

## Results on the held-out TEST split: before (skin-v1.0.0 as shipped) vs after (skin-v1.1.0)
Same 2 746 test images (574 healthy-skin patches included for both; the old model has no such class, it must pick one of its 19). "Tier accuracy" counts an uncertain answer as wrong (before: 0.849 of cases answered, accuracy on answered 0.846; after: all answered). Urgent = true red tier (melanoma, BCC, SCC) shown red; specificity = not-red shown not-red. Healthy false alarm = healthy patch shown amber or red. 95 % CIs (cluster bootstrap, 400 resamples) in `eval/metrics/skin.json` (`ci95`).

| Metric | Overall before | Overall after | PAD-UFES-20 (phone) before | after | SCIN (phone) before | after |
|---|---|---|---|---|---|---|
| n (urgent / healthy) | 2746 (392 / 574) | same | 517 (172 / 181) | same | 806 (3 / 393) | same |
| Tier accuracy | 0.718 | 0.702 | 0.505 | **0.870** | 0.730 | **0.911** |
| Urgent sensitivity | 0.602 | **0.923** | 0.733 | **0.919** | n/a (3 cases) | n/a (3 cases) |
| Urgent specificity | 0.978 | 0.838 | 0.948 | 0.890 | 1.000 | 0.998 |
| Urgent shown red or amber | 0.648 | 0.980 | 0.820 | 0.977 | 0.00 | 0.67 (2 of 3) |
| Healthy false alarm (amber/red) | 0.303 | **0.010** | 0.685 | **0.017** | 0.127 | **0.008** |
| Healthy not green (incl. uncertain) | 0.528 | 0.010 | 0.890 | 0.017 | 0.361 | 0.008 |
| 20-class top-1 | 0.558 | **0.752** | 0.429 | **0.779** | 0.174 | **0.654** |
| 20-class top-3 | 0.719 | **0.924** | 0.611 | **0.965** | 0.337 | **0.816** |
| 19 lesion classes only, top-1 / top-3 | 0.705 / 0.909 | 0.694 / 0.907 | 0.661 / 0.940 | 0.670 / 0.952 | 0.339 / 0.659 | 0.358 / 0.656 |

Honest reading:
- **Healthy skin:** shown as healthy-skin (top-1) for 97.0 % of test patches (PAD 98.3 %, SCIN 96.4 %); false alarm 1.0 % (CI 0.4 to 1.9 %, n = 574) vs 30 % before.
- **Urgent sensitivity 0.92 overall (CI 0.89 to 0.95) and 0.92 on PAD phone photos (CI 0.87 to 0.96)** (target on validation was 0.90; validation 0.903). **The price is specificity**: 0.84 overall, 0.89 on PAD: about 1 in 6 non-red cases is shown red, and nevus test images are shown amber or red in 52 % of cases (dermoscopy 53 %, 963 images; PAD phone photos 30 %, 30 images: 17 % amber, 13 % red). Tier accuracy overall falls from 0.72 to 0.70 because dermoscopy (HAM10000, not a phone domain) tier accuracy falls from 0.79 to 0.52 (red specificity 0.97 to 0.72); on phone-photo sources it rises strongly (PAD 0.51 to 0.87, SCIN 0.73 to 0.91, both mostly through the healthy class). Without the dark-spot rule overall tier accuracy is 0.757, PAD 0.878, SCIN 0.919 (the rule costs 0.8 to 5 points of "accuracy" for a safety property).
- 3 SCIN test cases are urgent: no conclusion on SCIN urgent sensitivity; PAD is the phone-photo indicator. Of the 6 melanoma / BCC / SCC photos tried in the browser, 2 (melanoma 966, SCC 633) were not shown red.
- Lesion-class discrimination is unchanged (19-class top-1 0.705 -> 0.694 overall; phone sources +1 point), as expected: the gain is the new class, the crop and the thresholds, not better disease recognition.
- Melanoma sensitivity (class) 0.69, BCC 0.77, SCC 0.17 (n = 30), AK 0.56, nevus 0.86, healthy-skin 0.97; inflammatory / infectious SCIN classes remain weak (0.07 to 0.55).
- By skin tone (SCIN dermatologist-estimated FST, tier accuracy, new model): FST1 0.88 (n 52), FST2 0.90 (256), FST3 0.89 (226), FST4 0.95 (131), FST5 0.92 (64), FST6 1.00 (n 2). Healthy false alarm per FST 0 to 6 % (FST1 1 of 16). FST 5-6 are still barely present.

## Browser check (Playwright, Chromium, real ONNX, upload path; full-resolution files, UI text of the result card)
| Image | App shows |
|---|---|
| (a) healthy-skin phone patch (PAD-UFES-20 test photo PAT_154_235_162, two patches) | green "ไม่พบสัญญาณน่ากังวล", closest match "ไม่พบรอยโรคที่ชัดเจน" 90 % and 92 % |
| (b) PAD melanoma test photo PAT_680_1289_182 | red "ควรพบแพทย์", melanoma 41 % (P(red) 0.41) |
| (b') PAD melanoma test photo PAT_966_1825_546 | amber "ควรระวัง", seborrheic keratosis 81 % (P(red) 0.07): a miss as red, amber by the look-alike guard |
| (c) PAD nevus test photos PAT_471_909_344 / _394 | amber "ควรระวัง", nevus 76 % / 73 % (guard: melanoma runner-up) |
| synthetic big black ellipse painted on a skin patch | amber (dark-spot rule); without the rule: green "vascular lesion 38 %" |
Other PAD test photos: actinic keratosis x2 amber, BCC x2 red, SCC 677 red, SCC 633 green (vascular lesion), SK 532 red, SK 1832 green (nevus). Not a sample: 14 photos, single draws.

## Known biases and limitations (v1.1 additions first; the v1.0.0 list below still applies)
- Sensitivity-first means many false alarms: 16 % of non-urgent cases are shown red on this test set, and half of the benign nevus images are shown amber or red (see above). Nothing here has been tested with real users, on phones, or prospectively.
- Healthy-skin is learned from patches of photos of people with lesions (PAD) or skin concerns (SCIN) and from auto-checked crops: unusual healthy skin (tattoos, scars, heavy freckling, sunburn, very dark skin in poor light, fingers/nails/lips) is not covered and may be called a lesion; tiny, plain, in-focus skin is the easiest case.
- The dark-spot rule is a hand-made heuristic (tested on a synthetic black ellipse and on PAD/SCIN statistics), not a learned feature. A black sticker, pen mark or shadow inside the box will also trigger amber.
- The camera path was verified with Chromium's fake camera and unit tests of the crop geometry, not on a real phone; the guide box assumes the preview is 4:3 with `object-fit: cover`.
- The quality check was relaxed (blur threshold 20 -> 6) because smooth healthy skin has a median Laplacian variance of 12 at 224 px; real defocus is still rejected, mild blur passes.
- Dermoscopy images (HAM10000) are still in training; their tier accuracy under the new rule is poor (0.52); the model is meant for phone photos.

---
# Previous release: skin-v1.0.0 (19 classes, whole-photo squash, argmax-of-tier-sums)

**Status:** real, trained baseline for a school project. **Not a medical device, not a diagnosis.** Every number below was measured on a held-out test split (never used for training, model selection, calibration or threshold choice). Tier assignments of the 11 SCIN-derived classes are provisional and **need clinician/teacher review**.

## Intended use
Screening aid inside the Vitals 360 AI web app: a phone photo of a skin area is classified on-device (no upload). The app shows (1) a traffic-light tier, (2) "closest match" class names with percentages, always with the text "not a diagnosis". Intended users: village health volunteers and the public in Thailand, as a prompt to see a clinician. Out of scope: diagnosis, treatment decisions, dermoscopy, children, mucosa/nails/scalp, any use without a clinician in the loop.

## Model
- Backbone: torchvision MobileNetV3-Large, ImageNet-1k pretrained (IMAGENET1K_V1). Blocks 0-6 frozen, blocks 7-16 and a new linear head (960 -> 19) fine-tuned on CPU: AdamW, one-cycle LR, 8 epochs, label smoothing 0.05, dropout 0.3, sqrt-balanced class weights, BatchNorm frozen, light augmentation (h/v flips, +-2 cell shifts at the 28x28 feature level). Best epoch by validation macro-F1.
- Input: RGB, longest side <= 640 (done by the app), area-weighted squash resize to 224x224 (no crop), ImageNet mean/std, NCHW float32. Same code path in Python (`training/cpu/skin_common.py`) and TypeScript (`src/inference/skin-preprocess.ts`).
- Output: 19 logits. `probs = softmax(logits / T)`, T = 0.8767 (fitted by minimising NLL on the validation split). Tier probability = sum of class probabilities per tier; shown tier = argmax; **uncertain** if that probability < 0.66 (picked on validation: lowest threshold where the answered subset is >= 5 points more accurate while still answering >= 60 % of cases).
- File: `public/models/skin-v1.int8.onnx`, 3.8 MB (one graph: backbone + head, logits out). Quantisation: per-channel int8 **weights** (DequantizeLinear in the graph) for layers with >= 20 000 weights, activations stay fp32. Full static int8 (QDQ, calibrated) was tried first and rejected: top-1 agreement with fp32 only 75.7 % (Percentile calibration 85 %, Entropy 79 % on 300 val images). Tiny early layers stay fp32 because they dominated the error.

## Data (all public, legally usable for education/research)
| Dataset | Licence | Images used | Domain |
|---|---|---|---|
| HAM10000 (Tschandl 2018, Harvard Dataverse doi:10.7910/DVN/DBW86T) | CC BY-NC 4.0 (non-commercial; fine for a school project) | 10 015 | dermoscopy, Austria/Australia |
| PAD-UFES-20 (Pacheco 2020, Mendeley doi:10.17632/zr7vgbcyr2.1) | CC BY 4.0 | 2 298 | smartphone photos, Brazil, biopsy-confirmed |
| Google SCIN (HF `google/scin`) | SCIN Data Use License (attribution, no re-identification, health education/research) | 2 888 images kept of 10 000+ (cases with a clear dermatologist label) | smartphone photos by US internet users, diverse skin tones |

Total after filtering: 15 201 images. SCIN labels are **dermatologist opinions from photos** (up to 3 graders, weighted), not biopsy: they are weak labels. A SCIN case is kept only if its top-weighted condition has weight >= 0.5 and is strictly above the runner-up, and the condition is in the class map. Ungradable cases are dropped. Raw data is not in the repo.

### Class map
| Class | Tier | HAM10000 | PAD-UFES-20 | SCIN condition(s) |
|---|---|---|---|---|
| melanoma | red | mel | MEL | Melanoma (+ in situ) |
| basal-cell-carcinoma | red | bcc | BCC | Basal Cell Carcinoma |
| squamous-cell-carcinoma | red | - | SCC | SCC/SCCIS |
| actinic-keratosis | amber | akiec (also contains Bowen) | ACK | Actinic Keratosis |
| nevus | green | nv | NEV | Melanocytic Nevus |
| seborrheic-keratosis | green | bkl (also benign lichenoid keratosis / solar lentigo) | SEK | SK/ISK |
| dermatofibroma | green | df | - | Dermatofibroma |
| vascular-lesion | green | vasc | - | Hemangioma |
| eczema | green | - | - | Eczema, acute/chronic dermatitis NOS, infected eczema, seborrheic dermatitis |
| contact-dermatitis | green | - | - | Allergic / irritant contact dermatitis |
| urticaria | green | - | - | Urticaria |
| psoriasis | amber | - | - | Psoriasis |
| tinea | green | - | - | Tinea, tinea versicolor |
| acne | green | - | - | Acne |
| folliculitis | green | - | - | Folliculitis |
| insect-bite | green | - | - | Insect bite |
| herpes-zoster | amber | - | - | Herpes zoster |
| herpes-simplex | green | - | - | Herpes simplex |
| purpura | amber | - | - | Purpura, ecchymoses, pigmented purpuric eruption |

Dropped for too few SCIN cases after filtering (< 40): impetigo (23 cases), keratosis pilaris (21). Other SCIN conditions (drug rash, lichen planus, pityriasis rosea, warts, ...) are not in the class set: such photos will be forced into the nearest of the 19 classes, which is a known limitation. No class is "healthy skin": the model has never seen normal skin.

### Split
Group-level, fixed seed 42, scikit-learn `StratifiedGroupKFold(7)` stratified on class: fold 0 = test, fold 1 = val, rest = train. Groups: HAM10000 `lesion_id`, PAD-UFES-20 `patient_id`, SCIN `case_id` (all images of one case stay together). HAM10000 has no patient id, so lesion-level, not patient-level (a patient with several lesions can span splits). Asserted: no group in two splits. Sizes (images): train 10 857, val 2 172, test 2 172 (test: HAM10000 1 423, PAD-UFES-20 336, SCIN 413; 1 462 groups). The split table is `training/cpu/results/skin-v1-split.csv.gz`.

## Results on the held-out TEST split (int8 model as shipped)
95 % CIs: cluster bootstrap over lesions/patients/cases, 1 000 resamples.

**3-tier level (what the traffic light shows, uncertain not counted):** accuracy 0.861 (CI 0.844 to 0.879); macro-F1 CI 0.716 to 0.777.

| Tier | n | Sensitivity | Specificity | F1 |
|---|---|---|---|---|
| green | 1581 | 0.937 | 0.765 | 0.925 |
| amber | 199 | 0.548 | 0.967 | 0.584 |
| red | 392 | 0.717 | 0.946 | 0.730 |

Confusion matrix (rows actual green/amber/red, columns predicted): [[1481, 36, 64], [57, 109, 33], [82, 29, 281]].

**Urgent-referral (red) sensitivity: 0.717 (CI 0.660 to 0.770), n = 392.** That means about 3 in 10 cancers in the test set are NOT shown as red. With the uncertain rule, 0.814 of true-red cases end up red or uncertain (both tell the user to see a clinician). The app must not be read as "green = safe".

Uncertain rule: answers 0.868 of cases; tier accuracy on answered cases 0.902.

**19-class level:** accuracy 0.705 (CI 0.682 to 0.728), top-3 accuracy 0.909, macro-sensitivity 0.427, macro-F1 0.424.

| Class | Tier | n (test) | Sensitivity | Specificity | F1 |
|---|---|---|---|---|---|
| melanoma | red | 166 | 0.61 | 0.965 | 0.60 |
| basal-cell-carcinoma | red | 196 | 0.75 | 0.971 | 0.73 |
| squamous-cell-carcinoma | red | 30 | 0.20 | 0.987 | 0.19 |
| actinic-keratosis | amber | 153 | 0.66 | 0.974 | 0.66 |
| nevus | green | 993 | 0.88 | 0.941 | 0.90 |
| seborrheic-keratosis | green | 192 | 0.69 | 0.958 | 0.65 |
| dermatofibroma | green | 17 | 0.41 | 0.997 | 0.45 |
| vascular-lesion | green | 22 | 0.86 | 0.998 | 0.83 |
| eczema | green | 119 | 0.50 | 0.962 | 0.46 |
| contact-dermatitis | green | 72 | 0.35 | 0.975 | 0.34 |
| urticaria | green | 49 | 0.37 | 0.992 | 0.42 |
| psoriasis | amber | 15 | 0.20 | 0.996 | 0.23 |
| tinea | green | 25 | 0.16 | 0.989 | 0.15 |
| acne | green | 12 | 0.33 | 0.995 | 0.30 |
| folliculitis | green | 27 | 0.11 | 0.992 | 0.13 |
| insect-bite | green | 39 | 0.38 | 0.983 | 0.33 |
| herpes-zoster | amber | 13 | 0.38 | 0.995 | 0.36 |
| herpes-simplex | green | 14 | 0.14 | 0.998 | 0.19 |
| purpura | amber | 18 | 0.11 | 0.994 | 0.12 |

Per-class numbers for classes with n < 30 are very noisy (one image moves sensitivity by 3-8 points). Inflammatory/infectious SCIN classes are weak (sensitivity 0.11 to 0.50): the "closest match" name for those is often wrong, and the app says so.

### By data source (tier level)
| Source | n | Tier accuracy | Red sensitivity |
|---|---|---|---|
| ham10000 | 1423 | 0.88 | 0.65 (n=217) |
| pad-ufes-20 | 336 | 0.79 | 0.81 (n=172) |
| scin | 413 | 0.86 | 0.00 (n=3) |

19-class accuracy by source: HAM10000 0.82, PAD-UFES-20 0.66, SCIN 0.34. SCIN has only 3 red cases in test, so red sensitivity on phone photos of non-PAD origin is unmeasured; PAD-UFES-20 (smartphone) red sensitivity is the better phone-photo indicator.

### By skin tone (tier accuracy; Fitzpatrick type)
SCIN: dermatologist-estimated Fitzpatrick (mode of up to 3 graders); PAD-UFES-20: recorded Fitzpatrick. HAM10000 has no skin-tone labels (almost all light skin).

SCIN (n images in test per type):

| FST | n | Tier accuracy | Red sensitivity |
|---|---|---|---|
| FST1 | 36 | 0.94 | n/a (no red cases) |
| FST2 | 152 | 0.83 | 0.00 (n=3) |
| FST3 | 124 | 0.85 | n/a (no red cases) |
| FST4 | 47 | 0.96 | n/a (no red cases) |
| FST5 | 33 | 0.94 | n/a (no red cases) |
| FST6 | 1 | 1.00 | n/a (no red cases) |

PAD-UFES-20:

| FST | n | Tier accuracy | Red sensitivity |
|---|---|---|---|
| FST1 | 25 | 0.72 | 0.89 (n=19) |
| FST2 | 121 | 0.74 | 0.79 (n=94) |
| FST3 | 59 | 0.78 | 0.85 (n=48) |
| FST4 | 20 | 0.85 | 0.73 (n=11) |
| FST5 | 2 | 0.50 | n/a (no red cases) |

Dark skin types (FST 5-6) are barely present (SCIN FST5 33, FST6 1; PAD FST5 2, FST6 none). No conclusion about performance on dark skin can be drawn from this test set. This is the largest known fairness gap.

### Quantisation parity (test images, n = 2172)
PyTorch vs ONNX fp32: max |logit diff| 2.1e-05, top-1 agreement 1.000. fp32 vs shipped int8-weights: max |diff| 0.246, mean |diff| 0.022, top-1 agreement 0.9922. Tier accuracy fp32 0.860 vs int8 0.861. In a real browser (onnxruntime-web 1.31.1 WASM) the logits match the Python int8 logits within 0.05 (e2e test `e2e/skin-model.spec.ts`). Desktop-CPU latency: 5.8 ms/image (model only; **not measured on a phone**).

## Known biases and limitations
- **Domain shift:** HAM10000 is dermoscopy, but the app takes ordinary phone photos. The model learns "dermoscopic look" cues; accuracy on phone photos is lower (PAD-UFES-20 tier accuracy 0.79 vs HAM10000 0.88). Source is also confounded with class (PAD-UFES-20 is 47 % cancers, SCIN has almost none), so the model may partly use "photo style" as a shortcut. Not measured here: other phones, lighting, blur, hair, jewellery, tattoos.
- **Skin tone:** training data is overwhelmingly light skin; SCIN adds darker types but few. Performance on FST 5-6 is unknown.
- **Weak labels** for the 11 SCIN classes (dermatologist opinion from a photo, up to 3 graders, many near-ties removed). Selection bias: only cases with a clear majority were kept, so real-world difficulty is higher.
- **Closed set:** anything not among the 19 classes (and healthy skin) is mapped to one of them.
- Class imbalance (nevus is 46 % of test). Class weights help rare classes but squamous-cell-carcinoma (sens 0.20, n = 30) and dermatofibroma (n = 17) are weak.
- Tier choices (e.g. psoriasis, herpes zoster, purpura amber; herpes simplex and urticaria green) were made by the team/AI and are provisional.
- Frozen early layers + 8 epochs on CPU is a baseline; GPU training with full fine-tuning, stronger augmentation and phone-domain data (see `docs/gpu-training-playbook.md`) should improve all numbers.
- Latency on phones, memory use and battery were not measured.

## Reproduce
See `docs/agents/training.md` (skin-v1.0.0: `skin_*.py`; skin-v1.1.0: `skin2_*.py`, `skin_pipe.py`, `skin_decide.py`).
