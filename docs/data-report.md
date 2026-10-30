# Vitals 360 AI: dataset report (Phase 2)

Checked online on **2026-10-30**. Every fact has a tag:
**[V]** read on the primary page or API (URL given), **[S]** only from a search snippet or a third-party page, **[U]** not verified (from memory or the paper; treat as a to-do).
Licences below are what the pages say today; this is not legal advice. For a school project (educational, non-commercial) CC BY-NC is fine. If the app is ever offered as a public service or sold, models trained on NC data need a lawyer's opinion, because the weights are derived from NC material.

## 0. Short answer (what to train on)

| Task | Train on | Evaluate on | Do not use |
|---|---|---|---|
| Skin, 3 tiers | PAD-UFES-20 (phone photos) + HAM10000 (+ ISIC 2019 only if you drop HAM10000, see leakage note) | held-out patient/lesion split; DDI and SCIN as external, per-skin-tone checks | DermNet (AI use prohibited), Fitzpatrick17k for training until URLs/labels are cleaned |
| Audio, 4 classes | ICBHI 2017 (normal/wheeze/crackle, stethoscope) + COUGHVID expert-labelled coughs (phone) ; SPRSound as an optional second stethoscope source | group-level split, reported per source | Coswara/Virufy for these labels (COVID labels only) |
| Camera respiration | no training: classical signal processing (already in `src/inference/resp.ts`) | COHFACE (needs a request), or a small team-recorded set with a reference count | UBFC-rPPG/PURE/MMPD for respiration (their ground truth is pulse, not breathing) |
| Fusion | hand-set or tiny logistic regression | see section 6: it cannot be validated with public data | claiming a measured multimodal gain |

The two biggest honest problems: (1) **domain gap**: skin public sets are mostly dermoscopy, audio sets are mostly stethoscope, while the app uses a phone camera and phone microphone; (2) **no paired multimodal data exists**, so the project hypothesis (multimodal beats unimodal) cannot be tested on public data.

## 1. Skin image datasets (ranked for this project)

| # | Dataset | Size | Licence | Classes / labels | Access | Direct download route |
|---|---|---|---|---|---|---|
| 1 | **PAD-UFES-20** (phone photos) | 2,298 images, 1,641 lesions, 1,373 patients [V] | CC BY 4.0 [V] | 6: BCC, SCC (incl. Bowen's), MEL, ACK (actinic keratosis), NEV, SEK. All 3 cancers biopsy-proven; the others ~58% biopsy-proven overall, rest clinical consensus [V] | open | Official: https://data.mendeley.com/datasets/zr7vgbcyr2/1 (Download All; `metadata.csv` is 316 KB [V]). HF mirror (unofficial, ungated, has `metadata.csv` + `all_images/imgs_part_*/*.png`) `SalmaneExploring/pad-ufes-20` [V]. Kaggle copy `mahdavi1202/skin-cancer` (third-party upload, exists per search [S]) |
| 2 | **HAM10000** | 10,015 dermoscopic images; images zip 1.37 GB + 1.40 GB [V] | **CC BY-NC 4.0** (Harvard Dataverse terms text) [V] | 7: nv, mel, bkl, bcc, akiec, vasc, df; >50% histopathology, rest follow-up/consensus/confocal [S] | open | https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/DBW86T (version 4, files listed via API [V]); Kaggle `kmader/skin-cancer-mnist-ham10000` (page exists [V]); metadata has `lesion_id` (use as the grouping key) [U] |
| 3 | **ISIC challenge sets** | 2019: 25,331 train (zip 9.77 GB [V]); 2020: 33,126; 2024 SLICE-3D: 401,059 (NC) or 217,477 permissive (CC BY) [V] | 2016-17 CC-0, 2018 task 3 / 2019 / 2020 / 2024 CC BY-NC, 2024 "Permissive" CC BY [V] | 2019: MEL, NV, BCC, AK, BKL, DF, VASC, SCC, UNK [U]; 2024: binary malignant/benign [V] | open | https://challenge.isic-archive.com/data/ ; direct S3 e.g. `https://isic-archive.s3.amazonaws.com/challenges/2019/ISIC_2019_Training_Input.zip` (HTTP 200 [V]); Kaggle `andrewmvd/isic-2019` (exists [V]) |
| 4 | **SCIN** (Google, phone photos, US volunteers) | 5,000+ contributions, 10,000+ images [V] | "SCIN Data Use License" (not CC): worldwide royalty-free, attribution, **no re-identification**, no stated commercial ban [V]; HF card says `license: other` [V] | dermatologist differential (weighted) labels; ~89% allergic/infectious/inflammatory among the 2,835 labelled [V]; Fitzpatrick (estimated + self-reported) and Monk tone [V] | open | GCS bucket `gs://dx-scin-public-data` [V]; HF `google/scin` (ungated, 26 parquet shards [V]); schema in `dataset_schema.md` of https://github.com/google-research-datasets/scin ; 15 duplicate images, 48 gradable cases without label, 1 missing file [V] |
| 5 | **Fitzpatrick17k** | 16,577 images, 114 conditions [V] | images CC BY-NC-SA 3.0 (from Atlas Dermatologico and DermaAmin) [V] | 114 conditions + Fitzpatrick type 1-6 per image (Scale AI + Centaur Labs raters; the authors call labelling skin type on diseased skin subjective) [V] | **CSV of URLs only**; some links are dead, authors give a form to request all images [V] | https://github.com/mattgroh/fitzpatrick17k . The HF `ZYXue/Fitzpatrick_17k` is a 5k-row VQA derivative, not the dataset [V]: do not use |
| 6 | **DDI** (Diverse Dermatology Images, Stanford) | 656 images, 570 patients, Fitzpatrick I-VI, pathology-confirmed [V] | Stanford Research Use Agreement: personal, **non-commercial research only; no redistribution, no sharing the link, no derivative works** [V] | benign/malignant + disease, skin tone group (I-II, III-IV, V-VI) [U] | **register + accept agreement** at Stanford AIMI portal [V] | https://ddi-dataset.github.io/ . Use only as a local, never-committed external test set |
| 7 | **DermNet** | ~20k+ images [U] | Free watermarked images: CC BY-NC-ND 4.0 and **"strictly prohibited" to download them to train or test AI**; paid licences for AI datasets [S] | 23 super-classes / 600+ conditions [U] | licence needed | https://dermnetnz.org/image-licence (page returned 404/503 to the fetch tool; read via search snippet [S]). **Excluded.** The original brief's 22 classes look DermNet-derived: we only reuse the class idea, not the images |

### Biases and quality notes
- **HAM10000**: dermoscopy (not phone), light skin, Austria and Australia (Vienna + Queensland) [U]; 67% of images are nevi; no patient id, so a few patients appear in several lesion groups (small leak when splitting by `lesion_id`).
- **ISIC 2019 contains HAM10000** (it is the BCN20000 + HAM10000 + MSK union) [U]: never combine both without de-duplicating by image id, or the test set leaks.
- **PAD-UFES-20**: Brazil, low-income patients of one university programme, mostly Fitzpatrick I-III (the CSV has a Fitzpatrick column; counts per type: compute at run time, printed by the notebook) [U]; only 6 classes; many ACK/BCC.
- **SCIN**: US, self-selected volunteers from search ads, labels are weak (dermatologist opinions from photos, not biopsy); good for common inflammatory conditions and skin-tone spread.
- **Fitzpatrick17k**: atlas images (clean, textbook), label noise reported in follow-up work [U], mislabelled skin types likely; for evaluation of skin-tone gaps use with caution.
- **Thailand**: none of these contain Thai patients as far as we could find. Thai skin is typically Fitzpatrick III-V [U]; HAM10000/PAD are mostly I-III. Expect a gap and say so in the app's model card.

## 2. Respiratory audio datasets

| # | Dataset | Size | Licence | Labels | Device / country | Access / download |
|---|---|---|---|---|---|---|
| 1 | **ICBHI 2017** | 920 recordings, 126 subjects, 5.5 h, 6,898 cycles: 1,864 crackle, 886 wheeze, 506 both, 3,642 normal [S] | "freely available for research" (official site) [S]; no machine-readable licence found; commercial use unverified [U] | per-cycle start/end + crackle (0/1) + wheeze (0/1) in `.txt`; patient diagnosis CSV [S] | Portugal + Greece, chest stethoscopes (Meditron, Littmann, AKG mic) [S/U], 4 kHz-44.1 kHz mixed [U] | official https://bhichallenge.med.auth.gr/ICBHI_2017_Challenge (page 503 at check time); zip URL `https://bhichallenge.med.auth.gr/sites/default/files/ICBHI_final_database/ICBHI_final_database.zip` answers HTTP 200 [V]; Kaggle `vbookshelf/respiratory-sound-database` (exists [V]); HF mirrors exist (e.g. `okchang95/ICBHI2017-raw`) [V] |
| 2 | **SPRSound** (SJTU paediatric) | 2,683 records, 9,089 events, 292 children [S] (BioCAS 2022 train 1,949 + test 734 files; 2023-25 test-only sets) [V] | **CC BY 4.0** on the repo [V] | record level: Normal, CAS, DAS, CAS&DAS, Poor Quality; event level: Normal, Rhonchi, Wheeze, Stridor, Coarse Crackle, Fine Crackle, Wheeze+Crackle, with ms timestamps; 11 paediatric physicians [V/S] | Shanghai Children's Medical Center, stethoscope, **children only** | https://github.com/SJTU-YONGFU-RESEARCH-GRP/SPRSound (clone). HF `Treza12/sprsound-classification` exists (unofficial) [V] |
| 3 | **COUGHVID** | v1.0: 951 MB zip, >20,000 crowdsourced cough recordings, >2,000 labelled by pulmonologists [V]; v3.0 zip 2.30 GB [V] | **CC BY 4.0** [V] | per-recording COVID status (self-reported), cough-detector score, SNR, expert labels (diagnosis, wheezing etc. per expert; column names to confirm in the notebook) [U] | worldwide phone microphones, uneven quality, self-reported | v1.0: https://zenodo.org/records/4048312 ; v3.0 `https://zenodo.org/api/records/7024894` file `public_dataset_v3.zip` [V]; formats webm/ogg (needs ffmpeg) |
| 4 | **Coswara** | ~2.7k+ participants listed; each gives 9 sounds (breathing fast/slow, cough heavy/shallow, vowels, counting) [V] | **CC BY 4.0** [V] | COVID status (healthy/exposed/positive/recovered), comorbidities, quality 0-2 [V] | India (IISc), phone, crowd | https://github.com/iiscleap/Coswara-Data (`extract_data.py`); HF `szzs1693/coswara-data` (CC BY 4.0, parquet) [V]. **No lung-sound labels**: use only as extra "normal" background/augmentation data, not for wheeze/crackle |
| 5 | **Virufy** | clinical set: 121 segmented coughs from 16 patients [V]; crowdsourced subset size unstated | licence not stated on page [V] | PCR-based COVID label [V] | hospital (country unstated), multi-country crowd | https://github.com/virufy/virufy-data . Too small, wrong labels: **skip** |

### Audio honesty box
- Wheeze and crackle labels exist **only for stethoscope audio on the chest** (ICBHI, SPRSound). The app asks a phone microphone to listen to a cough/breath. We found **no public phone-recorded dataset with wheeze/crackle labels** other than the few COUGHVID recordings where experts flagged wheezing [U]. So any wheeze/crackle accuracy from ICBHI will probably be **optimistic for the phone**; report it per source and tell users.
- Class and device are partly confounded in our recipe (normal/wheeze/crackle come from stethoscopes, cough-abnormal from phones). The notebook mixes healthy COUGHVID coughs into `normal` and reports per-source metrics so a model that just recognises the microphone shows up.
- ICBHI cycles with both crackle and wheeze (7%) are dropped (no such class in the manifest).
- "cough-abnormal" comes from COUGHVID experts' diagnosis (lower respiratory infection, obstructive disease): a few hundred recordings at most [U].

## 3. Camera respiration and chest motion

| Dataset | What it is | Ground truth | Access | Use for us |
|---|---|---|---|---|
| **COHFACE** (Idiap) | 160 one-minute RGB videos, 40 subjects, 640x480, 20 Hz [V] | respiration signal recorded with Thought Technology / BioGraph Infiniti [V] (belt type unstated) | Zenodo record 4081054 is **restricted (request)** [V]; Idiap page states no licence [V] | best public candidate to validate breaths/min; request access (a school email and a short project description) |
| **UBFC-rPPG** | 50 shared videos (8 simple + 42 realistic) [V] | pulse oximeter PPG + heart rate only [V] | Google Drive on request via the authors; no licence stated [V] | heart-rate only; **cannot validate respiration** |
| **PURE** | 10 subjects, motion conditions [U] | pulse oximeter [U] | raw data must be requested from the authors [S] | same, pulse only |
| **MMPD** (Tsinghua) | 33 subjects x 20 recordings, ~11 h; 370 GB (full) or 48 GB (mini); Fitzpatrick 3-6 [V] | PPG waveform [V] | **signed agreement sent from a faculty member's institutional email**; students cannot request; academic use only, commercial banned [V] | pulse only, and unattainable for us as students |
| **MSPM** (arXiv 2402.02224) | multi-site pulse/respiration/BP dataset; chest-motion respiration MAE 1.09 breaths/min reported [V] | pulse ox + BP (respiration reference unclear) [V] | no download link found [V] | watch; unverified availability |

Conclusion: respiration stays **classical signal processing**, validated first on synthetic sinusoids (already unit-tested), then on a **small team-recorded set** (10-20 volunteers, 30 s each, a teammate counts breaths against a stopwatch, or a belt) and on COHFACE if access is granted. Recording volunteers needs informed consent (PDPA treats health data as sensitive) and parent/teacher sign-off for minors.

## 4. Chest X-ray (optional, advanced, out of scope for this run)
NIH ChestX-ray14 (open, ~112k images, labels text-mined from reports, noisy [U]); CheXpert (Stanford, registration + research agreement [U]); MIMIC-CXR (PhysioNet **credentialed access**: CITI training course and signed DUA, normally for adults affiliated with an institution; not realistic for an M.2 team) [U]. None verified online in this run. A phone cannot take a CXR; this would only be a research side-project. Recommendation: skip.

## 5. Class sets and mapping tables

### Why few classes
22 classes with ~12k images of which 67% are nevi means most classes would have a few dozen test cases; per-class sensitivity would have confidence intervals wider than the effect. Three tiers map directly to the one decision the app supports: *do nothing / see a clinician routinely / see a clinician soon*. Fewer classes also let us report **sensitivity for the dangerous tier** with usable error bars. The model never names a disease.

### 5.1 Skin: tiers (labels exactly as in `public/models/manifest.json`)
- `benign` (green): common and usually harmless; self-care, see a doctor if it changes.
- `needs-review` (amber): treatable, pre-cancerous or possibly serious; routine clinic visit in the coming weeks.
- `urgent-referral` (red): suspected skin cancer; see a clinician soon. Not "you have cancer".
- `uncertain`: max probability below the manifest threshold (set after calibration); not a training class.

**Original 22 classes -> tier**

| Original class | Tier | Note |
|---|---|---|
| normal | benign | |
| acne | benign | severe cystic acne is a clinic matter, the app cannot tell |
| eczema | benign | |
| rosacea | benign | |
| seborrheic keratosis | benign | benign, but resembles melanoma: mistakes here cost sensitivity |
| moles / nevi | benign | the main false-negative risk is melanoma looking like a nevus |
| warts | benign | |
| benign tumors | benign | |
| sun damage | benign | |
| vascular tumors | benign | |
| actinic keratosis | needs-review | pre-cancerous |
| bullous disease | needs-review | |
| candidiasis | needs-review | |
| drug eruption | needs-review | severe forms (fever, mucosa, blisters) are urgent: handled by the symptom checklist, not by the image |
| infestations / bites | needs-review | |
| lichen planus-type | needs-review | |
| lupus / connective tissue | needs-review | |
| psoriasis | needs-review | |
| tinea / fungal | needs-review | |
| vasculitis | needs-review | |
| skin cancer (melanoma, BCC, SCC) | urgent-referral | |
| unknown | (excluded from training) | maps to "uncertain" at inference |

**Dataset label -> tier (implemented in `training/kaggle/skin.ipynb`)**

| Dataset | benign | needs-review | urgent-referral | Dropped |
|---|---|---|---|---|
| HAM10000 | nv, bkl, df, vasc | akiec | mel, bcc | none |
| PAD-UFES-20 | NEV, SEK | ACK | BCC, SCC, MEL | none |
| ISIC 2019 | NV, BKL, DF, VASC | AK | MEL, BCC, SCC | UNK |
| ISIC 2024 SLICE-3D | benign | (none) | malignant | indeterminate |
| DDI | `malignant == False` and disease in the benign list above | `malignant == False` and disease in the needs-review list | `malignant == True` | unclear disease names [U] |
| SCIN | map the top-weighted dermatologist condition through the 22-class table; keep only cases with confidence >= a threshold | same | cancers only | cases without a label |
| Fitzpatrick17k | 3-way partition "benign" + "non-neoplastic" conditions in the benign list | non-neoplastic conditions in the needs-review list | "malignant" partition (BCC, SCC, melanoma, Kaposi sarcoma, mycosis fungoides ...) | conditions that fit none |

SCIN/DDI/Fitzpatrick17k rows are a recipe, not code in this run; the exact condition lists must be built from the downloaded CSV and double-checked by a clinician or teacher before use [U].

Trade-offs: BKL/DF/VASC are grouped as benign although a few lesions need biopsy. AKIEC contains Bowen's disease (in-situ SCC) so a mistake between amber and red is clinically soft. We prefer an over-cautious operating point: the threshold is picked on validation, and the app also shows amber when the model is unsure between tiers (planned UX, not implemented here).

### 5.2 Audio: 4 classes (labels exactly as in the manifest)
| Source label | Class | Tier |
|---|---|---|
| ICBHI cycle (crackle 0, wheeze 0) | normal | green |
| ICBHI cycle (wheeze 1, crackle 0) | wheeze | amber |
| ICBHI cycle (crackle 1, wheeze 0) | crackle | amber |
| ICBHI cycle (both) | dropped | |
| SPRSound event: Wheeze | wheeze | amber |
| SPRSound event: Rhonchi | undecided: low-pitched continuous sound, ask a clinician; dropped until then | |
| SPRSound event: Fine/Coarse Crackle | crackle | amber |
| SPRSound event: Wheeze+Crackle, Stridor, Poor Quality | dropped | |
| COUGHVID majority expert `healthy_cough` | normal | green |
| COUGHVID majority expert `lower_infection` or `obstructive_disease` | cough-abnormal | amber |
| COUGHVID >= 2 experts flag wheezing | wheeze | amber |
| COUGHVID `upper_infection`, `COVID-19`, no majority | dropped | |
| Coswara, Virufy | not used | |

No audio class is red by design (matches the manifest).

## 6. Fairness plan

Goal: show whether the skin model works equally across skin tones, **where labels exist**, and say plainly where they do not.

1. **Groups**: Fitzpatrick I-II, III-IV, V-VI (the DDI grouping). Sources with a tone label: PAD-UFES-20 (`fitspatrick` column), SCIN (estimated and self-reported FST, Monk tone), Fitzpatrick17k, DDI. HAM10000: no tone label, state "not assessable".
2. **Metrics per group**: sensitivity and specificity of `urgent-referral` (primary), macro-F1 and accuracy (secondary), with Wilson 95% confidence intervals; groups with n < 20-30 are printed as `too_few`, not as numbers. The notebook already writes `by_fitzpatrick_PAD_only` and `by_source` into `report_skin.json`.
3. **External checks**: after choosing the model on validation, run once on DDI and a SCIN subset (local only; DDI forbids redistribution). Report the gap **I-II vs V-VI**. Flag any urgent-referral sensitivity gap > 10 percentage points.
4. **Gate**: if the gap is flagged, the model card and the in-app text say "less reliable on darker skin tones" and the abstain threshold for those cases is raised; we do not hide the number.
5. **Limits stated plainly**: no Thai data; tone labels are subjective (even for dermatologists on photos); camera/lighting differences are confounded with tone in these datasets; audio has no demographic fairness analysis beyond age/sex in COUGHVID and ICBHI.

## 7. The fusion problem and how to evaluate it honestly

**Problem.** The proposal's hypothesis is "multimodal beats unimodal". Fusion needs the *same person* to have a skin photo, a respiratory sound and a breathing video with a shared ground truth. We found **no public dataset with paired skin + audio + video**, and the three unimodal tasks do not even share a disease target (skin lesion type, lung sound type, breaths per minute). Pairing a skin image of patient A with a cough of patient B is not a patient.

**What not to do.** Randomly pair test samples and call the fused accuracy "measured multimodal performance". If the fused label is defined from the unimodal labels (e.g. worst tier), the fusion rule reproduces the label by construction: the result is circular, and if modalities are paired independently there is nothing for fusion to learn.

**Recommended evaluation (in order of honesty):**
1. **Simulation, clearly labelled SIMULATED.** Build pseudo-patients by pairing test cases of different modalities under an explicit generative assumption (for example the clinical risk tier is latent; each modality's output is drawn from its measured confusion matrix given that tier). This checks that the fusion code behaves (monotone in evidence, handles a missing or uncertain modality) and shows how much gain *would* be possible if modalities were conditionally independent. It cannot prove a real gain. The dashboard file for it must set `"isSample": true` or carry "SIMULATED" in `description`.
2. **Component tests on the real unimodal metrics**: report each unimodal model on its own held-out set, same split rules.
3. **A small prospective pilot** (the only way to test the hypothesis): consenting volunteers or a partner clinic (รพ.สต. / hospital), each doing all captures once, with a clinician's assessment as reference. Needs school/ethics approval and PDPA-compliant consent; realistic n is tens, so report confidence intervals and call it a pilot. Healthy volunteers alone give specificity but no sensitivity.
4. **Report the hypothesis as untested** on the dashboard if step 3 does not happen. The app's fusion output stays "low/moderate/high *indicative* risk" built from a transparent rule (`src/inference/fusion-weights.json`, flagged untrained).

## 8. What was not verified

- Kaggle copy `mahdavi1202/skin-cancer` contents and layout (only search snippet); the notebook locates files by pattern and prints what it found.
- ICBHI licence text and device list; official site returned 503.
- COUGHVID expert-label column names (`diagnosis_1..4`, `wheezing_1..4`); the notebook asserts them at run time.
- Fitzpatrick type distributions of PAD-UFES-20; SCIN and DDI per-condition mappings.
- Thai skin-tone statement, HAM10000 geography, ISIC-2019 / HAM10000 overlap (from memory of the papers).
- DermNet terms read via a search snippet only (the page fetch failed).

## Sources (all checked 2026-10-30)
- ISIC challenge data: https://challenge.isic-archive.com/data/
- HAM10000 Dataverse API/terms: https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/DBW86T
- PAD-UFES-20: https://data.mendeley.com/datasets/zr7vgbcyr2/1
- Fitzpatrick17k: https://github.com/mattgroh/fitzpatrick17k
- DDI: https://ddi-dataset.github.io/
- SCIN: https://github.com/google-research-datasets/scin , licence https://raw.githubusercontent.com/google-research-datasets/scin/main/LICENSE , HF https://huggingface.co/datasets/google/scin
- DermNet licence (snippet): https://dermnetnz.org/image-licence
- ICBHI: https://bhichallenge.med.auth.gr/ICBHI_2017_Challenge
- SPRSound: https://github.com/SJTU-YONGFU-RESEARCH-GRP/SPRSound
- COUGHVID: https://zenodo.org/records/4048312 , https://zenodo.org/records/7024894
- Coswara: https://github.com/iiscleap/Coswara-Data
- Virufy: https://github.com/virufy/virufy-data
- COHFACE: https://www.idiap.ch/en/scientific-research/data/cohface , https://zenodo.org/records/4081054
- UBFC-rPPG: https://sites.google.com/view/ybenezeth/ubfcrppg
- MMPD: https://github.com/McJackTang/MMPD_rPPG_dataset
- MSPM: https://arxiv.org/abs/2402.02224
