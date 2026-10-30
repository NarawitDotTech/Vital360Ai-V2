# Cell sources for skin.ipynb. ("md"|"code", text). Magic lines (!pip) are allowed; build_notebooks.py skips them in syntax checks.
CELLS = [
("md", r'''# Vitals 360 AI - skin model (3 tiers: benign / needs-review / urgent-referral)

Fine-tunes an ImageNet-pretrained **MobileNetV3-Large** (or EfficientNet-Lite0) on HAM10000 (+ PAD-UFES-20 phone photos if attached), calibrates it, exports **ONNX**, quantises to **int8**, checks **parity**, writes **dashboard metrics JSON** and a **manifest patch**.

**Before you run** (Kaggle): Settings -> Accelerator **GPU T4 x2 or P100**, Internet **On** (needs phone verification), Persistence: *Files only*. Add data (+ Add Input): `kmader/skin-cancer-mnist-ham10000` (required) and a PAD-UFES-20 copy such as `mahdavi1202/skin-cancer` (optional but recommended: real smartphone photos). Then *Run All*. First run: set `SMOKE = True` in cell 3 to check the whole pipeline in ~5 min.

NOT a medical device. Metrics from this notebook are only valid for the held-out split it prints. Read `docs/data-report.md` for dataset limits.

Runtimes below are ESTIMATES, not measured (no GPU was available when this was written).'''),
("code", r'''!pip install -q onnx onnxruntime onnxscript onnxconverter-common tensorboard'''),
("md", "Helper modules (the same files live in `training/kaggle/common/` in the repo)."),
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
("md", "## 1. Config"),
("code", r'''import json, random, io, time, glob, shutil, zipfile
from pathlib import Path
import numpy as np, pandas as pd, torch, torch.nn as nn
from common.seed import set_seed, worker_init

SMOKE = False            # True = tiny dry run (300 images, 2 epochs)
SEED = 42
ARCH = "mobilenet_v3_large"   # or "efficientnet_lite0" (needs timm)
EPOCHS = 2 if SMOKE else 15
BATCH = 64
IMG = 224                # MUST equal manifest onnx.imageSize / ort.ts default
CLASSES = ["benign", "needs-review", "urgent-referral"]   # MUST equal public/models/manifest.json skin labels
VERSION = "skin-v1-" + time.strftime("%Y%m%d")
WORK = Path("/kaggle/working"); OUT = WORK / "export"; CKPT = WORK / "ckpt_skin"; CACHE = WORK / "cache"
for d in (OUT, CKPT, CACHE): d.mkdir(parents=True, exist_ok=True)
set_seed(SEED)
DEV = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", DEV, torch.cuda.get_device_name(0) if DEV == "cuda" else "(no GPU: Settings -> Accelerator)")
print("torch", torch.__version__)'''),
("md", r'''## 2. Find the data and map labels to 3 tiers

Mapping (see `docs/data-report.md` section 4 for the rationale): benign = common, usually harmless; needs-review = see a clinician in the next weeks; urgent-referral = suspected skin cancer, see a clinician soon.'''),
("code", r'''HAM_MAP = {"nv": 0, "bkl": 0, "df": 0, "vasc": 0, "akiec": 1, "bcc": 2, "mel": 2}
PAD_MAP = {"NEV": 0, "SEK": 0, "ACK": 1, "BCC": 2, "SCC": 2, "MEL": 2}

def index_images(root, exts):
    return {p.name.lower(): p for e in exts for p in Path(root).rglob("*" + e)}

rows = []
ham_csv = next(iter(glob.glob("/kaggle/input/**/HAM10000_metadata*.csv", recursive=True)), None)
if ham_csv:
    imgs = index_images(Path(ham_csv).parent, [".jpg"])
    df = pd.read_csv(ham_csv)
    for r in df.itertuples():
        p = imgs.get(f"{r.image_id}.jpg".lower())
        if p is not None and r.dx in HAM_MAP:
            rows.append(dict(path=str(p), y=HAM_MAP[r.dx], group="ham_" + r.lesion_id, source="HAM10000", fitz=None, raw=r.dx))
    print("HAM10000 images:", sum(r["source"] == "HAM10000" for r in rows))
else:
    raise SystemExit("HAM10000 not found: + Add Input -> kmader/skin-cancer-mnist-ham10000")

pad_csv = None
for c in glob.glob("/kaggle/input/**/*.csv", recursive=True):
    try:
        cols = pd.read_csv(c, nrows=1).columns.str.lower()
    except Exception:
        continue
    if "diagnostic" in cols and "img_id" in cols:
        pad_csv = c; break
if pad_csv:
    imgs = index_images(Path(pad_csv).parent, [".png", ".jpg"])
    df = pd.read_csv(pad_csv); df.columns = [c.lower() for c in df.columns]
    fcol = next((c for c in df.columns if c.startswith("fitz")), None)   # the column is spelled "fitspatrick" in the original CSV
    for r in df.to_dict("records"):
        p = imgs.get(str(r["img_id"]).lower())
        if p is not None and r["diagnostic"] in PAD_MAP:
            f = r.get(fcol) if fcol else None
            rows.append(dict(path=str(p), y=PAD_MAP[r["diagnostic"]], group="pad_" + str(r["patient_id"]), source="PAD-UFES-20",
                             fitz=None if f is None or pd.isna(f) else int(f), raw=r["diagnostic"]))
    print("PAD-UFES-20 images:", sum(r["source"] == "PAD-UFES-20" for r in rows))
else:
    print("PAD-UFES-20 not attached: training on dermoscopy only, which will transfer poorly to phone photos.")

data = pd.DataFrame(rows)
if SMOKE: data = data.groupby("source", group_keys=False).sample(n=150, random_state=SEED, replace=True).drop_duplicates("path").reset_index(drop=True)
print(data.groupby(["source", "y"]).size().unstack(fill_value=0).rename(columns=dict(enumerate(CLASSES))))'''),
("md", r'''## 3. Group-level split (fixed seed)

Groups = `lesion_id` for HAM10000 (it has no patient id: two lesions of one patient can land in different splits, a known small leak) and `patient_id` for PAD-UFES-20. About 80/10/10 train/val/test. The test fold is touched only in cells 9-11.'''),
("code", r'''from sklearn.model_selection import StratifiedGroupKFold
k = 10 if not SMOKE else 5
folds = list(StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=SEED).split(data, data.y, data.group))
split = np.array(["train"] * len(data), dtype=object)
split[folds[0][1]] = "test"; split[folds[1][1]] = "val"
data["split"] = split
for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
    assert not set(data[data.split == a].group) & set(data[data.split == b].group), f"group leak {a}/{b}"
data.to_csv(WORK / "skin_splits.csv", index=False)
Y = data.y.values
print(data.groupby(["split", "y"]).size().unstack(fill_value=0).rename(columns=dict(enumerate(CLASSES))))'''),
("md", r'''## 4. Cache images as 224x224 uint8

The app **squashes the whole photo to 224x224 with no centre crop** (`ort.ts` skinTensor), so training does the same (aspect ratio is NOT preserved). ~12k images: about 5-10 min (estimate), resumable.'''),
("code", r'''from PIL import Image
from multiprocessing import Pool

def load224(p):
    try:
        return np.asarray(Image.open(p).convert("RGB").resize((IMG, IMG), Image.BILINEAR), dtype=np.uint8)
    except Exception as e:
        print("bad image", p, e); return np.zeros((IMG, IMG, 3), np.uint8)

cache_f = CACHE / ("skin224_smoke.npy" if SMOKE else "skin224.npy")
if cache_f.exists() and np.load(cache_f, mmap_mode="r").shape[0] == len(data):
    print("cache hit")
else:
    with Pool(os.cpu_count()) as pool:
        arr = np.stack(pool.map(load224, data.path.tolist(), chunksize=64))
    np.save(cache_f, arr); print("cached", arr.shape)
ARR = np.load(cache_f, mmap_mode="r")'''),
("md", "## 5. Datasets with phone-like augmentation"),
("code", r'''from torchvision.transforms import v2
import torch.nn.functional as F

class PhoneLike:
    """Cheap emulation of phone-camera variation: white balance/exposure, soft focus, JPEG, sensor noise."""
    def __call__(self, x):                                   # x uint8 CHW
        if random.random() < 0.6:
            g = torch.empty(3, 1, 1).uniform_(0.88, 1.12) * random.uniform(0.65, 1.35)
            x = (x.float() * g).clamp(0, 255).to(torch.uint8)
        if random.random() < 0.3:
            s = random.choice([96, 128, 160])
            x = F.interpolate(F.interpolate(x[None].float(), size=(s, s), mode="bilinear", antialias=True), size=(IMG, IMG), mode="bilinear")[0].clamp(0, 255).to(torch.uint8)
        if random.random() < 0.5:
            buf = io.BytesIO(); v2.functional.to_pil_image(x).save(buf, "JPEG", quality=random.randint(30, 92)); buf.seek(0)
            x = torch.from_numpy(np.asarray(Image.open(buf).convert("RGB"))).permute(2, 0, 1).contiguous()
        if random.random() < 0.3:
            x = (x.float() + torch.randn_like(x.float()) * random.uniform(2, 8)).clamp(0, 255).to(torch.uint8)
        return x

TRAIN_TF = v2.Compose([
    v2.RandomResizedCrop(IMG, scale=(0.55, 1.0), ratio=(0.8, 1.25), antialias=True),
    v2.RandomHorizontalFlip(), v2.RandomVerticalFlip(),
    v2.RandomApply([v2.RandomRotation(180)], p=0.5),
    v2.ColorJitter(0.35, 0.35, 0.35, 0.04),
    v2.RandomApply([v2.GaussianBlur(5, (0.1, 2.0))], p=0.3),
    PhoneLike(),
])

class SkinDS(torch.utils.data.Dataset):
    def __init__(self, idx, train): self.idx, self.train = np.asarray(idx), train
    def __len__(self): return len(self.idx)
    def __getitem__(self, i):
        j = self.idx[i]
        x = torch.from_numpy(np.ascontiguousarray(ARR[j])).permute(2, 0, 1)
        return (TRAIN_TF(x) if self.train else x), int(Y[j])

MEAN = torch.tensor([0.485, 0.456, 0.406], device=DEV).view(1, 3, 1, 1)   # == ort.ts skinTensor
STD = torch.tensor([0.229, 0.224, 0.225], device=DEV).view(1, 3, 1, 1)
def prep(x, train):
    return ((x.float() / 255 - MEAN) / STD).contiguous(memory_format=torch.channels_last)

def loader(split_name, train):
    idx = np.where(data.split.values == split_name)[0]
    return torch.utils.data.DataLoader(SkinDS(idx, train), batch_size=BATCH, shuffle=train, drop_last=train,
                                       num_workers=min(4, os.cpu_count()), pin_memory=True, worker_init_fn=worker_init,
                                       persistent_workers=True)
train_dl, val_dl = loader("train", True), loader("val", False)
print(len(train_dl.dataset), len(val_dl.dataset))'''),
("md", "## 6. Model, class weights, optimiser"),
("code", r'''def build(arch):
    if arch == "mobilenet_v3_large":
        from torchvision.models import mobilenet_v3_large, MobileNet_V3_Large_Weights
        m = mobilenet_v3_large(weights=MobileNet_V3_Large_Weights.IMAGENET1K_V2)
        m.classifier[-1] = nn.Linear(m.classifier[-1].in_features, len(CLASSES))
        groups = [{"params": m.features.parameters(), "lr": 4e-4}, {"params": m.classifier.parameters(), "lr": 1.5e-3}]
    else:
        import timm
        m = timm.create_model("tf_efficientnet_lite0.in1k", pretrained=True, num_classes=len(CLASSES))
        head = list(m.get_classifier().parameters()); hid = {id(p) for p in head}
        groups = [{"params": [p for p in m.parameters() if id(p) not in hid], "lr": 4e-4}, {"params": head, "lr": 1.5e-3}]
    return m, groups

model, groups = build(ARCH)
model = model.to(DEV).to(memory_format=torch.channels_last)
ytr = data.y.values[data.split.values == "train"]
cnt = np.bincount(ytr, minlength=len(CLASSES)); cw = (len(ytr) / (len(CLASSES) * cnt)) ** 0.5   # sqrt-inverse-frequency
print("train counts", cnt, "class weights", cw.round(2))
crit = nn.CrossEntropyLoss(weight=torch.tensor(cw, dtype=torch.float32, device=DEV), label_smoothing=0.05)
opt = torch.optim.AdamW(groups, weight_decay=1e-2)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
print(sum(p.numel() for p in model.parameters()) / 1e6, "M params")'''),
("md", r'''## 7. Train (checkpoint every epoch, resumable)

~1-2 min/epoch on a T4/P100 (estimate) -> 15 epochs about 15-30 min. **If the session dies:** reopen, *Run All* again; cell below resumes from `/kaggle/working/ckpt_skin/last.pt` (Files persistence must be on, or re-add the saved output as an input and copy `last.pt` back). TensorBoard event files are in `/kaggle/working/runs/skin` (download and run `tensorboard --logdir runs` locally).'''),
("code", r'''from torch.utils.tensorboard import SummaryWriter
from common.train import fit, predict
writer = SummaryWriter(str(WORK / "runs" / "skin"))
best = fit(model, train_dl, val_dl, crit, opt, sched, EPOCHS, CKPT, prep, len(CLASSES), writer=writer, patience=6)
writer.close(); print("best val macro-F1:", round(best, 4))'''),
("md", r'''## 8. Calibration on the validation set (temperature scaling + abstain threshold)

The app computes `softmax(logits / T)` and answers "uncertain" when max prob < threshold, so both numbers go into `manifest.json`.'''),
("code", r'''from common.calib import softmax, fit_temperature, ece, nll, coverage_table, pick_threshold
test_dl = loader("test", False)
zv, yv = predict(model, val_dl, prep)
T = fit_temperature(zv, yv)
th_rows = coverage_table(softmax(zv, T), yv)
THRESH = pick_threshold(th_rows, min_coverage=0.6, target_acc=0.85)
cal = {"temperature": T, "threshold": THRESH, "val_nll_before": nll(zv, yv), "val_nll_after": nll(zv, yv, T),
       "val_ece_before": ece(softmax(zv), yv), "val_ece_after": ece(softmax(zv, T), yv), "coverage_table_val": th_rows}
print({k: v for k, v in cal.items() if k != "coverage_table_val"})'''),
("md", "## 9. Export ONNX (fp32) and check parity against PyTorch on test images"),
("code", r'''from common.export import *
model = model.cpu().float().eval()
FP32 = OUT / "skin-fp32.onnx"
export_onnx(model, torch.zeros(1, 3, IMG, IMG), FP32)
te_idx = np.where(data.split.values == "test")[0]
Xte = np.stack([((np.asarray(ARR[j]).astype(np.float32) / 255 - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])).transpose(2, 0, 1) for j in te_idx]).astype(np.float32)
yte = data.y.values[te_idx]
with torch.no_grad(): zt_torch = np.concatenate([model(torch.from_numpy(Xte[i:i + 64])).numpy() for i in range(0, len(Xte), 64)])
zt_fp32 = run_onnx(FP32, Xte)
par_fp32 = parity_report(zt_torch, zt_fp32); print("fp32 parity", par_fp32)
assert par_fp32["max_abs_diff"] < 1e-3, "ONNX export does not match PyTorch"'''),
("md", "## 10. int8 (and fp16) variants, pick the smallest that passes the gates"),
("code", r'''from common.metrics import confusion, summary
f1_of = lambda z: summary(confusion(yte, z.argmax(1), len(CLASSES)))[0]["f1"]
f1_fp32 = f1_of(zt_fp32)
cands = {"fp32": (FP32, zt_fp32, par_fp32)}
tr_idx = np.where(data.split.values == "train")[0][:200]
calib_x = np.stack([((np.asarray(ARR[j]).astype(np.float32) / 255 - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])).transpose(2, 0, 1) for j in tr_idx]).astype(np.float32)
INT8 = OUT / "skin-int8.onnx"
try:
    quantize_int8(FP32, INT8, calib_x); z8 = run_onnx(INT8, Xte); cands["int8"] = (INT8, z8, parity_report(zt_torch, z8))
except Exception as e: print("int8 failed:", e)
FP16 = OUT / "skin-fp16.onnx"
if to_fp16(FP32, FP16):
    try: z16 = run_onnx(FP16, Xte); cands["fp16"] = (FP16, z16, parity_report(zt_torch, z16))
    except Exception as e: print("fp16 run failed (ORT CPU may lack fp16 kernels):", e)
LIMIT = 15_000_000
print(f"{'variant':6} {'bytes':>10} {'top1 agree':>10} {'macroF1':>8}")
ok = []
for n, (p, z, pr) in cands.items():
    sz = os.path.getsize(p); f1 = f1_of(z); print(f"{n:6} {sz:10d} {pr['top1_agreement']:10.4f} {f1:8.4f}")
    if sz <= LIMIT and (n == "fp32" or (pr["top1_agreement"] >= 0.97 and f1 >= f1_fp32 - 0.02)): ok.append((sz, n))
choice = min(ok)[1] if ok else "fp32"
if not ok: print("WARNING: nothing under 15 MB passes the gates; shipping fp32 anyway. Try EfficientNet-Lite0 or more QAT-like calibration.")
SHIP_SRC, z_ship, _ = cands[choice]
SHIP = OUT / "skin-v1.onnx"; shutil.copy(SHIP_SRC, SHIP)
print("shipping:", choice, os.path.getsize(SHIP), "bytes")'''),
("md", r'''## 11. Metrics on the held-out TEST split, using the file that will actually ship

`latencyMs` here is ONNX Runtime on the Kaggle CPU, **not a phone**. Re-measure on a mid-range Android before quoting it.'''),
("code", r'''from common.metrics import dashboard_entry, dashboard_file, subgroup_report
lat = latency_ms(SHIP, Xte[:1])
te = data.iloc[te_idx].reset_index(drop=True)
pred = z_ship.argmax(1)
desc = (f"Held-out group-level split (seed {SEED}), n={len(te)} test images; sources: " +
        ", ".join(f"{k} {v}" for k, v in te.source.value_counts().items()) +
        ". Labels collapsed to 3 tiers. Latency = ONNX Runtime CPU on Kaggle, not a phone.")
entry = dashboard_entry("Skin image", CLASSES, yte, pred, lat, os.path.getsize(SHIP), desc)
dump(dashboard_file("skin", entry, f"Real measured run {VERSION}. Not a diagnosis; see docs/data-report.md for limits."), OUT / "metrics_skin.json")
P = softmax(z_ship, T); conf = P.max(1); ans = conf >= THRESH
report = {"version": VERSION, "arch": ARCH, "choice": choice, "seed": SEED, "epochs": EPOCHS, "calibration": cal,
          "test_ece_after": ece(P, yte), "abstain": {"coverage": float(ans.mean()), "accuracy_answered": float((pred[ans] == yte[ans]).mean()) if ans.any() else None},
          "parity": {k: v[2] for k, v in cands.items()},
          "by_source": subgroup_report(te.source.values, yte, pred, CLASSES),
          "by_fitzpatrick_PAD_only": subgroup_report(te.fitz.fillna(-1).astype(int).values, yte, pred, CLASSES, min_n=20),
          "note": "Fitzpatrick groups with n<20 are flagged too_few; -1 = no label (all HAM10000). Brazilian PAD-UFES-20 is mostly Fitzpatrick I-III."}
dump(report, OUT / "report_skin.json")
patch = manifest_patch("skin", VERSION, SHIP, CLASSES, T, THRESH, extra_onnx={"input": "input", "output": "logits", "imageSize": IMG})
dump(patch, OUT / "manifest_patch_skin.json"); print(json.dumps(patch, indent=1)); print(json.dumps(entry, indent=1)[:1500])'''),
("md", r'''## 12. Bundle and download

Download `skin_bundle.zip` from the right-hand *Output* panel (or *Save Version -> Save & Run All (Commit)*, which keeps `/kaggle/working`).
Then, in the repo: copy `skin-v1.onnx` to `public/models/`, merge `manifest_patch_skin.json` into the skin entry of `public/models/manifest.json` (keep `tiers`), and copy `metrics_skin.json` to `eval/metrics/`. See `docs/gpu-training-playbook.md` section 7.'''),
("code", r'''with zipfile.ZipFile(WORK / "skin_bundle.zip", "w", zipfile.ZIP_DEFLATED) as z:
    for f in ["skin-v1.onnx", "metrics_skin.json", "report_skin.json", "manifest_patch_skin.json"]: z.write(OUT / f, f)
    z.write(WORK / "skin_splits.csv", "skin_splits.csv")
print("done:", (WORK / "skin_bundle.zip").stat().st_size, "bytes")'''),
]
