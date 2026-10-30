"""skin-v1.1: fine-tune MobileNetV3-Large blocks 7..16 + a 20-class head (19 skin classes + healthy-skin) on CPU, starting from the skin-v1
weights, with online phone-like augmentation (crop/scale jitter, flips/90deg turns, blur, JPEG, colour / white balance, lighting gradient,
noise). PAD-UFES-20 (smartphone) and SCIN are up-sampled, HAM10000 (dermoscopy) slightly down-sampled (nevus strongly). Blocks 0..6 stay frozen (BN frozen).
Usage: python skin2_train.py CACHE2 OLD_RUN_DIR RUN_DIR [epochs]"""
import copy, io, json, sys, threading, queue, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as Fn
from PIL import Image, ImageFilter
from scipy.special import softmax
from skin_calib import fit_temperature
from skin_classes import NAMES20 as NAMES
from skin_common import MEAN, STD
from skin_features import backbone
from skin_decide import TIER_IDX, decide, tier_sums

torch.set_num_threads(4)
K, NC, SEED = 7, len(NAMES), 42
SRC_W = {("pad-ufes-20", False): 3.0, ("scin", False): 2.0, ("ham10000", False): 0.8, ("pad-ufes-20", True): 1.5, ("scin", True): 1.0}
HAM_NEVUS_W = 0.3  # HAM10000 is dermoscopy: only the (huge) nevus group is strongly down-sampled; rare cancer classes keep most of their images
EPOCH_DRAWS = 14000


def augment(a: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """uint8 256x256x3 -> uint8 224x224x3 with phone-like perturbations."""
    im = Image.fromarray(a)
    s = a.shape[0]
    area = rng.uniform(0.45, 1.0) * s * s
    ar = np.exp(rng.uniform(np.log(0.8), np.log(1.25)))
    w, h = int(min(s, round(np.sqrt(area * ar)))), int(min(s, round(np.sqrt(area / ar))))
    x, y = int(rng.integers(0, s - w + 1)), int(rng.integers(0, s - h + 1))
    im = im.crop((x, y, x + w, y + h)).resize((224, 224), Image.BILINEAR, reducing_gap=2.0 if w > 448 else None)
    if rng.random() < 0.5: im = im.transpose(Image.FLIP_LEFT_RIGHT)
    if rng.random() < 0.5: im = im.transpose(Image.FLIP_TOP_BOTTOM)
    if rng.random() < 0.5: im = im.transpose(Image.ROTATE_90)
    if rng.random() < 0.3: im = im.rotate(float(rng.uniform(-20, 20)), Image.BILINEAR, fillcolor=None)
    if rng.random() < 0.35: im = im.filter(ImageFilter.GaussianBlur(float(rng.uniform(0.5, 2.2))))  # focus / motion softness
    x = np.asarray(im).astype(np.float32)
    # colour: white balance gains, brightness, contrast, saturation, gamma
    x *= rng.uniform(0.88, 1.12, 3).astype(np.float32)
    x *= rng.uniform(0.7, 1.3)
    m = x.mean()
    x = (x - m) * rng.uniform(0.75, 1.3) + m
    g = x.mean(2, keepdims=True)
    x = (x - g) * rng.uniform(0.7, 1.35) + g
    x = 255.0 * np.clip(x / 255.0, 0, 1) ** rng.uniform(0.8, 1.25)
    if rng.random() < 0.45:  # uneven lighting: linear gradient (+ optional vignette)
        ang = rng.uniform(0, 2 * np.pi)
        yy, xx = np.mgrid[-1:1:224j, -1:1:224j]
        grad = 1 + rng.uniform(0.1, 0.3) * (np.cos(ang) * xx + np.sin(ang) * yy)
        if rng.random() < 0.4: grad *= 1 - rng.uniform(0.1, 0.35) * (xx ** 2 + yy ** 2) / 2
        x *= grad[..., None].astype(np.float32)
    if rng.random() < 0.3: x += rng.normal(0, rng.uniform(2, 7), x.shape).astype(np.float32)
    x = np.clip(x, 0, 255).astype(np.uint8)
    if rng.random() < 0.5:  # JPEG re-compression
        buf = io.BytesIO(); Image.fromarray(x).save(buf, "JPEG", quality=int(rng.integers(35, 92))); buf.seek(0)
        x = np.asarray(Image.open(buf).convert("RGB"))
    return x


def to_tensor_batch(u8: np.ndarray) -> torch.Tensor:
    x = (torch.from_numpy(u8).float() / 255.0 - torch.from_numpy(MEAN)) / torch.from_numpy(STD)
    return x.permute(0, 3, 1, 2).contiguous()


class Top(nn.Module):
    def __init__(self, bb, K=K):
        super().__init__()
        self.blocks = nn.Sequential(*list(bb.features)[K:])
        self.pool = bb.avgpool
        self.drop = nn.Dropout(0.3)
        self.head = nn.Linear(960, NC)

    def forward(self, x):
        return self.head(self.drop(torch.flatten(self.pool(self.blocks(x)), 1)))


def bn_eval(m):
    for l in m.modules():
        if isinstance(l, nn.BatchNorm2d): l.eval()


@torch.no_grad()
def stage1_of(front, u8, bs=128):
    out = np.zeros((len(u8), 40, 28, 28), dtype=np.float16)
    for i in range(0, len(u8), bs):
        out[i:i + bs] = front(to_tensor_batch(np.asarray(u8[i:i + bs]))).numpy().astype(np.float16)
    return out


@torch.no_grad()
def logits_of(model, S1):
    model.eval()
    return np.concatenate([model(torch.from_numpy(S1[i:i + 128]).float()).numpy() for i in range(0, len(S1), 128)])


def macro_f1(y, p):
    f = []
    for c in range(NC):
        tp = ((p == c) & (y == c)).sum(); fp = ((p == c) & (y != c)).sum(); fn = ((p != c) & (y == c)).sum()
        if (y == c).any(): f.append(2 * tp / max(2 * tp + fp + fn, 1))
    return float(np.mean(f))


def main(c2: Path, old_run: Path, run: Path, epochs=8):
    torch.manual_seed(SEED); rng = np.random.default_rng(SEED)
    run.mkdir(parents=True, exist_ok=True)
    idx = pd.read_csv(c2 / "index2.csv")
    idx.to_csv(run / "split.csv", index=False)
    y = idx["cls"].map({n: i for i, n in enumerate(NAMES)}).to_numpy()
    E = np.load(c2 / "eval224.npy", mmap_mode="r"); T = np.load(c2 / "train256.npy", mmap_mode="r")
    tr, va = np.where(idx.split == "train")[0], np.where(idx.split == "val")[0]
    hs = (idx.cls == "healthy-skin").to_numpy()
    w_s = np.array([HAM_NEVUS_W if (s == "ham10000" and c == "nevus") else SRC_W[(s, h)] for s, h, c in zip(idx.source, hs, idx.cls)])
    # sampling distribution and sqrt-balanced class weights over what is actually drawn
    p_s = w_s[tr] / w_s[tr].sum()
    eff = np.bincount(y[tr], weights=p_s, minlength=NC) * EPOCH_DRAWS
    cw = torch.tensor((eff.sum() / (NC * np.maximum(eff, 1))) ** 0.5, dtype=torch.float32)
    print("effective draws per class:", dict(zip(NAMES, eff.round().astype(int))), flush=True)
    # model: v1 backbone + head (+1 row for healthy-skin)
    bb = backbone(); bb.load_state_dict(torch.load(old_run / "backbone.pt"))
    front = nn.Sequential(*list(bb.features)[:K]).eval()
    model = Top(bb)
    h = np.load(old_run / "head.npz")
    with torch.no_grad():
        model.head.weight[:19] = torch.from_numpy(h["W"]); model.head.bias[:19] = torch.from_numpy(h["b"])
        model.head.weight[19:] = 0.0; model.head.bias[19:] = float(h["b"].min())
    t0 = time.time()
    S1va = stage1_of(front, E[va]); print("val stage1", S1va.shape, f"{time.time()-t0:.0f}s", flush=True)
    opt = torch.optim.AdamW([{"params": model.blocks.parameters(), "lr": 1.5e-4}, {"params": model.head.parameters(), "lr": 1.5e-3}], weight_decay=1e-2)
    steps_per = EPOCH_DRAWS // 64
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[1.5e-4, 1.5e-3], total_steps=epochs * steps_per, pct_start=0.15)
    pool = ThreadPoolExecutor(3)

    def make_batch(b, seed):
        r = np.random.default_rng(seed)
        return np.stack([augment(np.asarray(T[i]), r) for i in b])

    best = (-1, None, None)
    for ep in range(epochs):
        model.train(); bn_eval(model)
        draws = rng.choice(tr, size=steps_per * 64, p=p_s)
        q: "queue.Queue" = queue.Queue(maxsize=6)
        def producer():
            futs = []
            for s in range(steps_per):
                b = np.sort(draws[s * 64:(s + 1) * 64])
                futs.append((b, pool.submit(make_batch, b, int(rng.integers(0, 2 ** 31)))))
                while len(futs) > 5:
                    q.put(futs.pop(0))
            for f in futs: q.put(f)
        threading.Thread(target=producer, daemon=True).start()
        tl = 0.0
        for s in range(steps_per):
            b, fut = q.get()
            with torch.no_grad():
                s1 = front(to_tensor_batch(fut.result()))
            loss = Fn.cross_entropy(model(s1), torch.from_numpy(y[b]), weight=cw, label_smoothing=0.05)
            opt.zero_grad(); loss.backward(); opt.step(); sched.step(); tl += loss.item()
            if s % 50 == 0: print(f"  ep{ep} step {s}/{steps_per} loss {loss.item():.3f} ({time.time()-t0:.0f}s)", flush=True)
        lv = logits_of(model, S1va)
        pv = lv.argmax(1)
        f1 = macro_f1(y[va], pv)
        tacc = float((TIER_IDX[pv] == TIER_IDX[y[va]]).mean())
        hv = hs[va]
        print(f"epoch {ep} loss {tl/steps_per:.3f} val acc {(pv==y[va]).mean():.4f} macroF1 {f1:.4f} tier-acc {tacc:.4f} healthy-recall {(pv[hv]==NAMES.index('healthy-skin')).mean():.3f} ({time.time()-t0:.0f}s)", flush=True)
        if f1 > best[0]:
            best = (f1, copy.deepcopy(model.state_dict()), ep)
            torch.save(best[1], run / "top_best.pt")
    model.load_state_dict(best[1]); print("best epoch", best[2], flush=True)
    lv = logits_of(model, S1va)
    Tm = fit_temperature(lv, y[va])
    full = backbone()
    for i, l in enumerate(list(full.features)[K:]):
        l.load_state_dict(model.blocks[i].state_dict())
    torch.save(full.state_dict(), run / "backbone.pt")
    np.savez(run / "head.npz", W=model.head.weight.detach().numpy(), b=model.head.bias.detach().numpy(), T=Tm)
    json.dump(dict(method="finetune blocks 7-16 from skin-v1, online phone-like augmentation, +healthy-skin", classes=NAMES, epochs=epochs, best_epoch=best[2], T=Tm,
                   seed=SEED, sampling=str({f"{k[0]}{'-healthy' if k[1] else ''}": v for k, v in SRC_W.items()}), epoch_draws=EPOCH_DRAWS, train_seconds=time.time() - t0),
              open(run / "train_report.json", "w"), indent=1)
    print("T", Tm, flush=True)


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]), int(sys.argv[4]) if len(sys.argv) > 4 else 8)
