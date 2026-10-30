CELLS = [
("md", r'''# Vitals 360 AI - audio model (normal / wheeze / crackle / cough-abnormal)

Small CNN on **log-mel** features that match `src/inference/mel.ts` exactly (16 kHz, n_fft 1024, hop 320, 64 Slaney mels, `log(1e-10 + power)`, 501 frames = 10 s). Trains on ICBHI 2017 (stethoscope, cycle labels) + COUGHVID expert-labelled phone coughs, exports ONNX, checks parity, writes dashboard metrics and a mel parity fixture for the TypeScript test.

**Before you run** (Kaggle): Accelerator GPU (T4/P100; a CPU run also works for this small model but is slower), Internet **On**. Add input `vbookshelf/respiratory-sound-database` (ICBHI). COUGHVID is downloaded from Zenodo by cell 5 (951 MB). First run: `SMOKE = True`.

HONEST WARNING: ICBHI is chest-stethoscope audio, the app records phone-microphone audio. Wheeze/crackle labels that were measured on stethoscope data may not transfer to a phone at the mouth/chest. This is a screening *prototype*; see `docs/data-report.md` section 5. Runtimes are estimates, not measured.'''),
("code", r'''!pip install -q onnx onnxruntime onnxscript tensorboard soundfile librosa'''),
("code", r'''import os, sys
os.makedirs("common", exist_ok=True)
sys.path.insert(0, os.getcwd())'''),
("writefile_common", "__init__"),
("writefile_common", "seed"),
("writefile_common", "ckpt"),
("writefile_common", "calib"),
("writefile_common", "metrics"),
("writefile_common", "train"),
("writefile_common", "export"),
("writefile_common", "mel"),
("md", "## 1. Config"),
("code", r'''import json, random, time, glob, shutil, zipfile, re, subprocess
from collections import Counter
from pathlib import Path
import numpy as np, pandas as pd, torch, torch.nn as nn, librosa
from common.seed import set_seed, worker_init
from common.mel import LogMel, SR, CLIP, FRAMES, N_MELS, write_fixture

SMOKE = False
SEED = 42
EPOCHS = 2 if SMOKE else 40
BATCH = 64
CLASSES = ["normal", "wheeze", "crackle", "cough-abnormal"]   # MUST equal public/models/manifest.json audio labels
VERSION = "audio-v1-" + time.strftime("%Y%m%d")
WORK = Path("/kaggle/working"); OUT = WORK / "export"; CKPT = WORK / "ckpt_audio"; DATA = WORK / "data"
for d in (OUT, CKPT, DATA): d.mkdir(parents=True, exist_ok=True)
set_seed(SEED)
DEV = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", DEV, "| frames", FRAMES, "| clip samples", CLIP)'''),
("md", r'''## 2. ICBHI 2017 -> cycles

Each `.txt` row = `start end crackle wheeze`. Label: (0,0) normal, (1,0) crackle, (0,1) wheeze. **Cycles with both crackle and wheeze (~7%) are dropped** because the app has no "both" class. Group = patient id (first token of the file name).'''),
("code", r'''clips, meta = [], []   # clips: float32 mono 16 kHz arrays; meta: dict(y, group, source)
txts = [p for p in glob.glob("/kaggle/input/**/*.txt", recursive=True) if re.match(r"^\d{3}_\d+[a-z]\d_", os.path.basename(p))]
if not txts: raise SystemExit("ICBHI not found: + Add Input -> vbookshelf/respiratory-sound-database")
if SMOKE: txts = txts[:60]
for t in txts:
    wav = t[:-4] + ".wav"
    if not os.path.exists(wav): continue
    x, _ = librosa.load(wav, sr=SR, mono=True)
    pid = os.path.basename(t).split("_")[0]
    for line in open(t):
        s, e, c, w = line.split()[:4]; c, w = int(c), int(w)
        if c and w: continue
        seg = x[int(float(s) * SR): int(float(e) * SR)]
        if len(seg) < int(0.25 * SR): continue
        clips.append(seg.astype(np.float32)); meta.append(dict(y=1 if w else 2 if c else 0, group="icbhi_" + pid, source="ICBHI"))
print("ICBHI cycles:", len(clips), Counter(m["y"] for m in meta))'''),
("md", r'''## 3. COUGHVID (phone microphones, crowdsourced) with expert labels

Downloaded from Zenodo record 4048312 (CC BY 4.0, 951 MB). Only recordings with at least one pulmonologist label are used. Mapping: majority expert diagnosis `healthy_cough` -> normal; `lower_infection` or `obstructive_disease` -> cough-abnormal; >=2 experts flag wheezing -> wheeze; everything else (`upper_infection`, `COVID-19`, unclear) is skipped. Column names are asserted at runtime (they were not verified when this was written).'''),
("code", r'''z = DATA / "coughvid.zip"
if not (DATA / "public_dataset").exists():
    if not z.exists() or z.stat().st_size < 9e8:
        subprocess.run(["curl", "-L", "-C", "-", "-o", str(z), "https://zenodo.org/records/4048312/files/public_dataset.zip?download=1"], check=True)
    zipfile.ZipFile(z).extractall(DATA)
mcsv = next(iter(glob.glob(str(DATA / "**/metadata_compiled.csv"), recursive=True)))
cv = pd.read_csv(mcsv); print(list(cv.columns))
dcols = [c for c in cv.columns if re.fullmatch(r"diagnosis_\d", c)]; wcols = [c for c in cv.columns if re.fullmatch(r"wheezing_\d", c)]
assert dcols, "expected diagnosis_1..4 expert columns; check the printed column list and adapt"
def cv_label(r):
    d = [r[c] for c in dcols if isinstance(r[c], str)]
    if not d: return None
    if sum(bool(r[c]) is True and r[c] == r[c] for c in wcols) >= 2: return 1
    top, n = Counter(d).most_common(1)[0]
    if n * 2 <= len(d): return None
    return {"healthy_cough": 0, "lower_infection": 3, "obstructive_disease": 3}.get(top)
n0 = len(clips); base = Path(mcsv).parent
if "cough_detected" in cv: cv = cv[cv.cough_detected.fillna(0) >= 0.8]
if SMOKE: cv = cv.head(300)
for r in cv.to_dict("records"):
    y = cv_label(r)
    if y is None: continue
    f = next((base / (r["uuid"] + e) for e in (".webm", ".ogg", ".wav") if (base / (r["uuid"] + e)).exists()), None)
    if f is None: continue
    try: x, _ = librosa.load(str(f), sr=SR, mono=True)
    except Exception: continue
    if len(x) < int(0.25 * SR): continue
    clips.append(x[:CLIP].astype(np.float32)); meta.append(dict(y=y, group="cv_" + r["uuid"], source="COUGHVID"))
print("COUGHVID clips used:", len(clips) - n0)
md = pd.DataFrame(meta); print(md.groupby(["source", "y"]).size().unstack(fill_value=0).rename(columns=dict(enumerate(CLASSES))))'''),
("md", r'''## 4. Group-level split (patient for ICBHI, recording for COUGHVID: it has no user id)

~80/10/10. COUGHVID may contain several recordings of one person in different splits (known, small leak).'''),
("code", r'''from sklearn.model_selection import StratifiedGroupKFold
k = 10 if not SMOKE else 5
folds = list(StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=SEED).split(md, md.y, md.group))
sp = np.array(["train"] * len(md), dtype=object); sp[folds[0][1]] = "test"; sp[folds[1][1]] = "val"; md["split"] = sp
for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
    assert not set(md[md.split == a].group) & set(md[md.split == b].group), "group leak"
md.to_csv(WORK / "audio_splits.csv", index=False)
print(md.groupby(["split", "y"]).size().unstack(fill_value=0).rename(columns=dict(enumerate(CLASSES))))'''),
("md", r'''## 5. 10-second canvases + phone-like augmentation

The app feeds a continuous 10 s recording (501 frames). Training samples are single cycles/coughs, so each is placed on a 10 s canvas: random offset, and with p=0.5 (always at eval) tiled with short gaps to mimic repeated breaths/coughs. Augmentation: gain, noise floor, random band-limit (phone mic), then SpecAugment on the mel. Gain does not matter at inference because the model subtracts the per-clip mean of the log-mel.'''),
("code", r'''from scipy.signal import butter, sosfilt
def canvas(clip, rng, train):
    L = CLIP; clip = clip[:L]
    if len(clip) < L // 2 and (not train or rng.random() < 0.5):
        gap = np.zeros(int((rng.uniform(0.1, 0.6) if train else 0.3) * SR), np.float32)
        clip = np.tile(np.concatenate([clip, gap]), L // (len(clip) + len(gap)) + 2)[:L]
    off = int(rng.integers(0, L - len(clip) + 1)) if train else 0
    out = np.zeros(L, np.float32); out[off: off + len(clip)] = clip
    peak = np.abs(out).max() + 1e-8
    out = out / peak * (10 ** rng.uniform(-1.6, -0.1) if train else 0.5)
    if train and rng.random() < 0.5:
        lo, hi = rng.uniform(60, 300), rng.uniform(3500, 7500)
        out = sosfilt(butter(4, [lo, hi], btype="band", fs=SR, output="sos"), out).astype(np.float32)
    out += rng.standard_normal(L).astype(np.float32) * (rng.uniform(5e-5, 2e-3) if train else 5e-4)
    return out

class AudioDS(torch.utils.data.Dataset):
    def __init__(self, idx, train): self.idx, self.train = np.asarray(idx), train
    def __len__(self): return len(self.idx)
    def __getitem__(self, i):
        j = self.idx[i]
        rng = np.random.default_rng() if self.train else np.random.default_rng(int(j))
        return torch.from_numpy(canvas(clips[j], rng, self.train)), int(Y[j])
Y = md.y.values
LOGMEL = LogMel().to(DEV)
def prep(x, train):
    m = LOGMEL(x)                                # [B,1,501,64]
    if train:                                    # SpecAugment: 2 freq masks, 2 time masks
        for _ in range(2):
            f = random.randint(0, 8); f0 = random.randint(0, N_MELS - f); m[:, :, :, f0:f0 + f] = m.mean()
            t = random.randint(0, 40); t0 = random.randint(0, FRAMES - t); m[:, :, t0:t0 + t, :] = m.mean()
    return m
def loader(name, train):
    idx = np.where(md.split.values == name)[0]
    return torch.utils.data.DataLoader(AudioDS(idx, train), batch_size=BATCH, shuffle=train, drop_last=train, num_workers=min(4, os.cpu_count()), worker_init_fn=worker_init)
train_dl, val_dl = loader("train", True), loader("val", False)'''),
("md", r'''## 6. Model

Input = raw log-mel `[N,1,501,64]` exactly as `ort.ts` builds it. The model removes the per-clip mean and divides by a fixed std **inside the graph**, so the app does no normalisation.'''),
("code", r'''with torch.no_grad():
    sample = [prep(torch.from_numpy(canvas(clips[j], np.random.default_rng(j), True))[None].to(DEV), False) for j in np.where(md.split.values == "train")[0][:300]]
    STDV = float(torch.cat([s - s.mean() for s in sample]).std())
print("input std (after per-clip mean removal):", round(STDV, 3))

class Net(nn.Module):
    def __init__(self, k, std):
        super().__init__()
        self.register_buffer("std", torch.tensor(float(std)))
        def blk(i, o): return nn.Sequential(nn.Conv2d(i, o, 3, padding=1, bias=False), nn.BatchNorm2d(o), nn.ReLU(inplace=True),
                                            nn.Conv2d(o, o, 3, padding=1, bias=False), nn.BatchNorm2d(o), nn.ReLU(inplace=True), nn.MaxPool2d(2))
        self.f = nn.Sequential(blk(1, 16), blk(16, 32), blk(32, 64), blk(64, 128))
        self.drop = nn.Dropout(0.3); self.fc = nn.Linear(256, k)
    def forward(self, x):
        x = (x - x.mean(dim=(1, 2, 3), keepdim=True)) / self.std
        x = self.f(x)
        return self.fc(self.drop(torch.cat([x.mean(dim=(2, 3)), x.amax(dim=(2, 3))], 1)))

model = Net(len(CLASSES), STDV).to(DEV)
ytr = Y[md.split.values == "train"]; cnt = np.bincount(ytr, minlength=len(CLASSES))
cw = (len(ytr) / (len(CLASSES) * np.maximum(cnt, 1))) ** 0.5
print("train counts", cnt, "weights", cw.round(2), "| params", sum(p.numel() for p in model.parameters()))
crit = nn.CrossEntropyLoss(weight=torch.tensor(cw, dtype=torch.float32, device=DEV), label_smoothing=0.05)
opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-2)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)'''),
("md", r'''## 7. Train (resumable)

Estimate: ~15-40 s/epoch on a T4 (data loading bound), 40 epochs about 10-25 min. After a disconnect: Run All again, the loop resumes from `/kaggle/working/ckpt_audio/last.pt`. TensorBoard files: `/kaggle/working/runs/audio`.'''),
("code", r'''from torch.utils.tensorboard import SummaryWriter
from common.train import fit, predict
writer = SummaryWriter(str(WORK / "runs" / "audio"))
best = fit(model, train_dl, val_dl, crit, opt, sched, EPOCHS, CKPT, prep, len(CLASSES), writer=writer, patience=10)
writer.close(); print("best val macro-F1:", round(best, 4))'''),
("md", "## 8. Calibration on validation (temperature + abstain threshold)"),
("code", r'''from common.calib import softmax, fit_temperature, ece, nll, coverage_table, pick_threshold
zv, yv = predict(model, val_dl, prep)
T = fit_temperature(zv, yv)
rows = coverage_table(softmax(zv, T), yv, thresholds=np.arange(0.3, 0.9, 0.05))
THRESH = pick_threshold(rows, min_coverage=0.6, target_acc=0.8)
cal = {"temperature": T, "threshold": THRESH, "val_nll_before": nll(zv, yv), "val_nll_after": nll(zv, yv, T),
       "val_ece_before": ece(softmax(zv), yv), "val_ece_after": ece(softmax(zv, T), yv), "coverage_table_val": rows}
print({k: v for k, v in cal.items() if k != "coverage_table_val"})'''),
("md", "## 9. Export ONNX + parity (mel inputs from the test split)"),
("code", r'''from common.export import *
from common.metrics import confusion, summary, dashboard_entry, dashboard_file, subgroup_report
model = model.cpu().float().eval()
FP32 = OUT / "audio-fp32.onnx"
export_onnx(model, torch.zeros(1, 1, FRAMES, N_MELS), FP32)
te_idx = np.where(md.split.values == "test")[0]
tdl = loader("test", False); lm = LogMel()
Xte = np.concatenate([lm(w).numpy() for w, _ in tdl]).astype(np.float32); yte = np.concatenate([y.numpy() for _, y in tdl])
with torch.no_grad(): zt_torch = np.concatenate([model(torch.from_numpy(Xte[i:i + 64])).numpy() for i in range(0, len(Xte), 64)])
zt_fp32 = run_onnx(FP32, Xte); par = parity_report(zt_torch, zt_fp32); print("fp32 parity", par)
assert par["max_abs_diff"] < 1e-3, "ONNX export does not match PyTorch"'''),
("md", r'''## 10. Optional int8, choose artifact

The model is ~1 MB in fp32, far under the 15 MB budget, so fp32 is shipped unless int8 is both smaller and equal in quality (top-1 agreement >= 0.97, macro-F1 drop <= 0.02) and fp32 is over 4 MB.'''),
("code", r'''f1_of = lambda z: summary(confusion(yte, z.argmax(1), len(CLASSES)))[0]["f1"]
cands = {"fp32": (FP32, zt_fp32, par)}
INT8 = OUT / "audio-int8.onnx"
try:
    quantize_int8(FP32, INT8, Xte[:min(150, len(Xte))]); z8 = run_onnx(INT8, Xte); cands["int8"] = (INT8, z8, parity_report(zt_torch, z8))
except Exception as e: print("int8 skipped:", e)
for n, (p, z, pr) in cands.items(): print(n, os.path.getsize(p), pr["top1_agreement"], round(f1_of(z), 4))
choice = "fp32"
if "int8" in cands and os.path.getsize(FP32) > 4_000_000 and cands["int8"][2]["top1_agreement"] >= 0.97 and f1_of(cands["int8"][1]) >= f1_of(zt_fp32) - 0.02: choice = "int8"
SHIP = OUT / "audio-v1.onnx"; shutil.copy(cands[choice][0], SHIP); z_ship = cands[choice][1]
print("shipping", choice, os.path.getsize(SHIP), "bytes")'''),
("md", "## 11. Metrics, report, manifest patch, mel fixture"),
("code", r'''lat = latency_ms(SHIP, Xte[:1])
te = md.iloc[te_idx].reset_index(drop=True); pred = z_ship.argmax(1)
desc = (f"Held-out group-level split (seed {SEED}), n={len(te)} 10 s canvases built from single ICBHI cycles / COUGHVID coughs; "
        + ", ".join(f"{k} {v}" for k, v in te.source.value_counts().items()) + ". Cycles with both crackle+wheeze excluded. Latency = ONNX Runtime CPU on Kaggle, not a phone.")
entry = dashboard_entry("Audio", CLASSES, yte, pred, lat, os.path.getsize(SHIP), desc)
dump(dashboard_file("audio", entry, f"Real measured run {VERSION}. Stethoscope + phone-cough data mixed; see docs/data-report.md."), OUT / "metrics_audio.json")
P = softmax(z_ship, T); ans = P.max(1) >= THRESH
report = {"version": VERSION, "choice": choice, "seed": SEED, "calibration": cal, "test_ece_after": ece(P, yte),
          "abstain": {"coverage": float(ans.mean()), "accuracy_answered": float((pred[ans] == yte[ans]).mean()) if ans.any() else None},
          "parity": {k: v[2] for k, v in cands.items()}, "by_source": subgroup_report(te.source.values, yte, pred, CLASSES),
          "caveat": "ICBHI = stethoscope; COUGHVID = phone. Class and recording device are partly confounded; trust by_source, not the pooled number."}
dump(report, OUT / "report_audio.json")
dump(manifest_patch("audio", VERSION, SHIP, CLASSES, T, THRESH, extra_onnx={"input": "input", "output": "logits", "frames": FRAMES}), OUT / "manifest_patch_audio.json")
fx = write_fixture(OUT / "mel_fixture.json"); print("fixture frames", fx["nFrames"], "mels", fx["nMels"])
with zipfile.ZipFile(WORK / "audio_bundle.zip", "w", zipfile.ZIP_DEFLATED) as zf:
    for f in ["audio-v1.onnx", "metrics_audio.json", "report_audio.json", "manifest_patch_audio.json", "mel_fixture.json"]: zf.write(OUT / f, f)
print(json.dumps(entry, indent=1)[:1500])'''),
("md", r'''## 12. Next steps

Download `audio_bundle.zip`. Copy `audio-v1.onnx` to `public/models/`, merge `manifest_patch_audio.json` into the audio entry of `public/models/manifest.json` (keep `tiers`), copy `metrics_audio.json` to `eval/metrics/`, and add `mel_fixture.json` as a vitest fixture for `src/inference/mel.ts` (compare `logMel(samples, 16000)` with `expected`, atol ~5e-3). Details: `docs/gpu-training-playbook.md` section 7.'''),
]
