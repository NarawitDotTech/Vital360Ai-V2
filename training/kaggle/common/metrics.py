"""Metrics + the dashboard JSON (schema: eval/metrics/README.md, schemaVersion 1)."""
import datetime as _dt

import numpy as np


def confusion(y: np.ndarray, p: np.ndarray, k: int) -> np.ndarray:
    cm = np.zeros((k, k), dtype=int)
    for a, b in zip(y, p):
        cm[int(a), int(b)] += 1
    return cm


def per_class(cm: np.ndarray):
    """One-vs-rest sensitivity (recall), specificity, precision, f1 per class."""
    n = cm.sum()
    out = []
    for i in range(cm.shape[0]):
        tp = cm[i, i]
        fn = cm[i].sum() - tp
        fp = cm[:, i].sum() - tp
        tn = n - tp - fn - fp
        sens = tp / (tp + fn) if tp + fn else float("nan")
        spec = tn / (tn + fp) if tn + fp else float("nan")
        prec = tp / (tp + fp) if tp + fp else float("nan")
        f1 = 2 * prec * sens / (prec + sens) if (prec + sens) and not np.isnan(prec + sens) else float("nan")
        out.append({"sensitivity": sens, "specificity": spec, "precision": prec, "f1": f1, "support": int(tp + fn)})
    return out


def summary(cm: np.ndarray):
    pc = per_class(cm)
    mean = lambda key: float(np.nanmean([c[key] for c in pc]))  # noqa: E731
    return {"accuracy": float(np.trace(cm) / max(1, cm.sum())), "sensitivity": mean("sensitivity"),
            "specificity": mean("specificity"), "f1": mean("f1")}, pc


def _clean(x):
    return None if (isinstance(x, float) and np.isnan(x)) else round(float(x), 4)


def dashboard_entry(label, classes, y, pred, latency_ms, size_bytes, test_desc):
    cm = confusion(y, pred, len(classes))
    s, pc = summary(cm)
    return {
        "label": label, "classes": list(classes),
        "accuracy": _clean(s["accuracy"]), "sensitivity": _clean(s["sensitivity"]),
        "specificity": _clean(s["specificity"]), "f1": _clean(s["f1"]),
        "perClass": {c: {"sensitivity": _clean(pc[i]["sensitivity"]), "specificity": _clean(pc[i]["specificity"])}
                     for i, c in enumerate(classes)},
        "confusionMatrix": cm.tolist(), "latencyMs": round(float(latency_ms), 1),
        "sizeBytes": int(size_bytes), "testSetDescription": test_desc,
    }


def dashboard_file(model_id, entry, description):
    return {"schemaVersion": 1, "isSample": False,
            "generatedAt": _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "description": description, "models": {model_id: entry}}


def subgroup_report(groups, y, pred, classes, min_n: int = 30):
    """Per-subgroup (e.g. Fitzpatrick type, data source) accuracy and per-class sensitivity.
    Groups with fewer than min_n samples are reported with a 'too_few' flag instead of numbers."""
    groups = np.asarray(groups)
    out = {}
    for g in sorted(set(groups.tolist())):
        m = groups == g
        n = int(m.sum())
        if n < min_n:
            out[str(g)] = {"n": n, "too_few": True}
            continue
        cm = confusion(y[m], pred[m], len(classes))
        pc = per_class(cm)
        out[str(g)] = {"n": n, "accuracy": _clean(np.trace(cm) / n),
                       "sensitivity": {c: _clean(pc[i]["sensitivity"]) for i, c in enumerate(classes)},
                       "support": {c: pc[i]["support"] for i, c in enumerate(classes)}}
    return out
