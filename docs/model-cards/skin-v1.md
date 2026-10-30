# Model card: skin-v1 (19-class skin image screening baseline)

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
See `docs/agents/training.md`.
