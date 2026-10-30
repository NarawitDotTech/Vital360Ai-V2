"""Frozen-feature baseline: split by group, logistic-regression head, temperature scaling, threshold.
Usage: python skin_train.py CACHE_DIR OUT_DIR   (writes head.npz, split.csv, train_report.json)"""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
from scipy.optimize import minimize_scalar
from scipy.special import log_softmax, softmax
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from skin_metrics import CLASSES, bootstrap, confusion, summarize

SEED = 42


def make_split(idx: pd.DataFrame) -> np.ndarray:
    """Group-level (lesion for HAM, patient for PAD) stratified split: 7 folds -> fold0 test, fold1 val, rest train.
    Stratified on source*class so both domains are present in every split."""
    strat = idx["source"].str[:3] + idx["y"].astype(str)
    split = np.array(["train"] * len(idx), dtype=object)
    folds = list(StratifiedGroupKFold(n_splits=7, shuffle=True, random_state=SEED).split(idx, strat, idx["group"]))
    split[folds[0][1]] = "test"
    split[folds[1][1]] = "val"
    return split


def nll(logits, y, T):
    return -log_softmax(logits / T, axis=1)[np.arange(len(y)), y].mean()


def calibrate(lv, yv):
    """Temperature (min NLL on val) and uncertainty threshold from val logits."""
    T = float(minimize_scalar(lambda t: nll(lv, yv, t), bounds=(0.2, 10), method="bounded").x)
    pv = softmax(lv / T, axis=1)
    # threshold: lowest max-prob cut-off whose confident-subset val accuracy is >= overall + 0.05 while coverage >= 0.6
    base = float((pv.argmax(1) == yv).mean())
    thr = 0.0
    for t in np.arange(0.34, 0.95, 0.01):
        m = pv.max(1) >= t
        if m.mean() < 0.6:
            break
        if (pv.argmax(1)[m] == yv[m]).mean() >= base + 0.05:
            thr = float(round(t, 2)); break
    return T, (thr if thr > 0 else 0.5), base


def main(cache: Path, out: Path):
    out.mkdir(parents=True, exist_ok=True)
    idx = pd.read_csv(cache / "index.csv")
    split = make_split(idx)
    idx["split"] = split
    idx.to_csv(out / "split.csv", index=False)
    assert not (set(idx[idx.split == "train"].group) & set(idx[idx.split == "test"].group))
    assert not (set(idx[idx.split == "val"].group) & set(idx[idx.split == "test"].group))
    assert not (set(idx[idx.split == "train"].group) & set(idx[idx.split == "val"].group))
    y = idx["y"].to_numpy()
    F = {v: np.load(cache / f"feat_{v}.npy") for v in ("orig", "hflip", "vflip")}
    tr, va, te = (split == "train"), (split == "val"), (split == "test")
    Xtr = np.concatenate([F[v][tr] for v in F]); ytr = np.concatenate([y[tr]] * 3)
    sc = StandardScaler().fit(Xtr)
    best = None
    for C in (0.0003, 0.001, 0.003, 0.01, 0.03):
        clf = LogisticRegression(C=C, class_weight="balanced", max_iter=2000, random_state=SEED).fit(sc.transform(Xtr), ytr)
        lv = clf.decision_function(sc.transform(F["orig"][va]))
        s = summarize(confusion(y[va], lv.argmax(1)))
        score = s["f1"].mean()
        print(f"C={C} val macroF1={score:.4f} acc={s['accuracy']:.4f} nll={nll(lv, y[va], 1):.4f}", flush=True)
        if best is None or score > best[0]:
            best = (score, C, clf)
    _, C, clf = best
    lv = clf.decision_function(sc.transform(F["orig"][va]))
    T, thr, base = calibrate(lv, y[va])
    # fold standardisation into the linear head: logits = W x + b on raw pooled features
    W = clf.coef_ / sc.scale_[None, :]
    b = clf.intercept_ - (W * sc.mean_[None, :]).sum(1)
    np.savez(out / "head.npz", W=W.astype(np.float32), b=b.astype(np.float32), T=T, thr=thr)
    json.dump(dict(C=C, T=T, threshold=thr, val_acc=float(base), seed=SEED,
                   sizes={s: int((split == s).sum()) for s in ("train", "val", "test")}), open(out / "train_report.json", "w"), indent=1)
    print("C", C, "T", T, "thr", thr, "val acc", base)


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
