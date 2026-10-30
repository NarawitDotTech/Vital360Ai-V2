"""Metrics + bootstrap CIs for 3-class tiers."""
import numpy as np

CLASSES = ["benign", "needs-review", "urgent-referral"]


def confusion(y, p, k=3):
    cm = np.zeros((k, k), dtype=int)
    np.add.at(cm, (y, p), 1)
    return cm


def summarize(cm):
    n = cm.sum()
    tp = np.diag(cm).astype(float)
    fn = cm.sum(1) - tp
    fp = cm.sum(0) - tp
    tn = n - tp - fn - fp
    sens = np.divide(tp, tp + fn, out=np.zeros_like(tp), where=(tp + fn) > 0)
    spec = np.divide(tn, tn + fp, out=np.zeros_like(tn), where=(tn + fp) > 0)
    prec = np.divide(tp, tp + fp, out=np.zeros_like(tp), where=(tp + fp) > 0)
    f1 = np.divide(2 * prec * sens, prec + sens, out=np.zeros_like(tp), where=(prec + sens) > 0)
    return dict(accuracy=tp.sum() / n, sens=sens, spec=spec, f1=f1)


def bootstrap(y, p, groups, n_boot=1000, seed=0):
    """Cluster bootstrap over groups (patients/lesions). Returns {metric: (lo, hi)} 95% percentile CIs."""
    rng = np.random.default_rng(seed)
    ug, inv = np.unique(groups, return_inverse=True)
    idx_by_g = [np.where(inv == i)[0] for i in range(len(ug))]
    keys = {}
    for _ in range(n_boot):
        pick = rng.integers(0, len(ug), len(ug))
        idx = np.concatenate([idx_by_g[i] for i in pick])
        s = summarize(confusion(y[idx], p[idx]))
        vals = {"accuracy": s["accuracy"], "macro_sens": s["sens"].mean(), "macro_f1": s["f1"].mean()}
        for i, c in enumerate(CLASSES):
            vals[f"sens_{c}"] = s["sens"][i]
            vals[f"spec_{c}"] = s["spec"][i]
        for k, v in vals.items():
            keys.setdefault(k, []).append(v)
    return {k: (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))) for k, v in keys.items()}


def bootstrap_generic(groups, fn, n_boot=1000, seed=0):
    """Cluster bootstrap: fn(row_indices) -> {name: value}. Returns {name: [lo, hi]} (95% percentile, NaNs ignored)."""
    rng = np.random.default_rng(seed)
    ug, inv = np.unique(groups, return_inverse=True)
    idx_by_g = [np.where(inv == i)[0] for i in range(len(ug))]
    vals = {}
    for _ in range(n_boot):
        idx = np.concatenate([idx_by_g[i] for i in rng.integers(0, len(ug), len(ug))])
        for k, v in fn(idx).items():
            vals.setdefault(k, []).append(v)
    return {k: [float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5))] for k, v in vals.items()}
