"""(Also builds the split.) Fine-tune MobileNetV3-Large blocks 7..16 + a new linear head on CPU, using cached stage-1 activations (blocks 0..6 frozen).
Usage: python skin_finetune.py CACHE_DIR RUN_DIR   (needs cache/index_all.csv + images_all.npy from skin_build_index.py)"""
import copy, json, sys, time
from pathlib import Path
import numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as Fn
from skin_features import backbone, prep
from skin_metrics import confusion, summarize
from sklearn.model_selection import StratifiedGroupKFold
from skin_calib import TIER_IDX, fit_temperature, pick_threshold, tier_probs
from skin_classes import NAMES

SEED = 42
NC = len(NAMES)

torch.set_num_threads(4)
K = 7  # features[:K] frozen and cached
EPOCHS = 8


@torch.no_grad()
def stage1(cache: Path):
    p = cache / "stage1_all.npy"
    if p.exists():
        return np.load(p, mmap_mode="r")
    imgs = np.load(cache / "images_all.npy", mmap_mode="r")
    m = backbone().features[:K]
    out = np.zeros((3, len(imgs), 40, 28, 28), dtype=np.float16)
    for i in range(0, len(imgs), 64):
        x = prep(np.asarray(imgs[i:i + 64]))
        for v, f in enumerate((x, x.flip(3), x.flip(2))):
            out[v, i:i + 64] = m(f).numpy().astype(np.float16)
        if (i // 64) % 20 == 0:
            print("stage1", i, flush=True)
    np.save(p, out)
    return np.load(p, mmap_mode="r")


class Top(nn.Module):
    def __init__(self, bb):
        super().__init__()
        self.blocks = nn.Sequential(*list(bb.features)[K:])
        self.pool = bb.avgpool
        self.drop = nn.Dropout(0.3)
        self.head = nn.Linear(960, NC)

    def forward(self, x):
        return self.head(self.drop(torch.flatten(self.pool(self.blocks(x)), 1)))


def bn_eval(m):
    for l in m.modules():
        if isinstance(l, nn.BatchNorm2d):
            l.eval()


def make_split(sp):
    """Group-level (HAM lesion / PAD patient / SCIN case) split, stratified on class: 7 folds -> fold0 test, fold1 val, rest train."""
    split = np.array(["train"] * len(sp), dtype=object)
    folds = list(StratifiedGroupKFold(n_splits=7, shuffle=True, random_state=SEED).split(sp, sp["cls"], sp["group"]))
    split[folds[0][1]] = "test"
    split[folds[1][1]] = "val"
    return split


def logits_of(model, S1, idx):
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(idx), 128):
            out.append(model(torch.from_numpy(np.asarray(S1[0, idx[i:i + 128]])).float()).numpy())
    return np.concatenate(out)


def main(cache: Path, run: Path):
    torch.manual_seed(SEED); rng = np.random.default_rng(SEED)
    run.mkdir(parents=True, exist_ok=True)
    sp = pd.read_csv(cache / "index_all.csv")
    sp["split"] = make_split(sp)
    sp.to_csv(run / "split.csv", index=False)
    y = sp["cls"].map({n: i for i, n in enumerate(NAMES)}).to_numpy()
    for a, b in (("train", "test"), ("val", "test"), ("train", "val")):
        assert not (set(sp[sp.split == a].group) & set(sp[sp.split == b].group))
    tr, va = np.where(sp.split == "train")[0], np.where(sp.split == "val")[0]
    S1 = np.asarray(stage1(cache))
    bb = backbone(); model = Top(bb)
    cnt = np.bincount(y[tr], minlength=NC)
    w = torch.tensor((cnt.sum() / (NC * np.maximum(cnt, 1))) ** 0.5, dtype=torch.float32)  # sqrt-balanced class weights
    opt = torch.optim.AdamW([{"params": model.blocks.parameters(), "lr": 2e-4}, {"params": model.head.parameters(), "lr": 2e-3}], weight_decay=1e-2)
    steps = EPOCHS * (len(tr) // 64)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[2e-4, 2e-3], total_steps=steps, pct_start=0.15)
    best = (-1, None, None); t0 = time.time()
    for ep in range(EPOCHS):
        model.train(); bn_eval(model)
        perm = rng.permutation(tr)
        for i in range(0, len(perm) - 63, 64):
            b = perm[i:i + 64]
            v = rng.integers(0, 3, len(b))
            x = torch.from_numpy(S1[v, b]).float()
            dx, dy = rng.integers(-2, 3, 2)
            x = torch.roll(x, (int(dy), int(dx)), (2, 3))
            loss = Fn.cross_entropy(model(x), torch.from_numpy(y[b]), weight=w, label_smoothing=0.05)
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        lv = logits_of(model, S1, va)
        s = summarize(confusion(y[va], lv.argmax(1), NC))
        f1 = float(s["f1"][np.bincount(y[va], minlength=NC) > 0].mean())
        tacc = float((TIER_IDX[lv.argmax(1)] == TIER_IDX[y[va]]).mean())
        urg = (TIER_IDX[y[va]] == 2); usens = float((TIER_IDX[lv.argmax(1)][urg] == 2).mean())
        print(f"epoch {ep} loss {loss.item():.3f} val acc {s['accuracy']:.4f} macroF1 {f1:.4f} tier-acc {tacc:.4f} urgent-sens {usens:.3f} ({time.time()-t0:.0f}s)", flush=True)
        if f1 > best[0]:
            best = (f1, copy.deepcopy(model.state_dict()), ep)
    model.load_state_dict(best[1]); print("best epoch", best[2])
    lv = logits_of(model, S1, va)
    T = fit_temperature(lv, y[va])
    thr, base = pick_threshold(tier_probs(lv, T), TIER_IDX[y[va]])
    full = backbone()
    for i, l in enumerate(list(full.features)[K:]):
        l.load_state_dict(model.blocks[i].state_dict())
    torch.save(full.state_dict(), run / "backbone.pt")
    np.savez(run / "head.npz", W=model.head.weight.detach().numpy(), b=model.head.bias.detach().numpy(), T=T, thr=thr)
    json.dump(dict(method="finetune blocks 7-16", classes=NAMES, epochs=EPOCHS, best_epoch=best[2], T=T, threshold=thr, val_acc=base, seed=SEED,
                   train_seconds=time.time() - t0), open(run / "train_report.json", "w"), indent=1)
    print("T", T, "thr", thr, "val acc", base)


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
