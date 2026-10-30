import numpy as np
import torch


def softmax(z: np.ndarray, T: float = 1.0) -> np.ndarray:
    z = z / T
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def nll(logits: np.ndarray, y: np.ndarray, T: float = 1.0) -> float:
    p = softmax(logits, T)
    return float(-np.log(np.clip(p[np.arange(len(y)), y], 1e-12, 1)).mean())


def fit_temperature(logits: np.ndarray, y: np.ndarray) -> float:
    """Temperature scaling (Guo et al. 2017): minimise NLL over a single scalar T on the VALIDATION set."""
    z = torch.tensor(logits, dtype=torch.float64)
    t = torch.tensor(y, dtype=torch.long)
    log_t = torch.zeros(1, dtype=torch.float64, requires_grad=True)
    opt = torch.optim.LBFGS([log_t], lr=0.1, max_iter=200)

    def closure():
        opt.zero_grad()
        loss = torch.nn.functional.cross_entropy(z / log_t.exp(), t)
        loss.backward()
        return loss

    opt.step(closure)
    return float(log_t.exp().clamp(0.05, 20.0).item())


def ece(probs: np.ndarray, y: np.ndarray, bins: int = 10) -> float:
    conf = probs.max(1)
    pred = probs.argmax(1)
    edges = np.linspace(0, 1, bins + 1)
    out = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            out += m.mean() * abs((pred[m] == y[m]).mean() - conf[m].mean())
    return float(out)


def coverage_table(probs: np.ndarray, y: np.ndarray, thresholds=None):
    """For each confidence threshold: fraction of cases answered, accuracy on those answered."""
    thresholds = thresholds if thresholds is not None else np.arange(0.3, 0.96, 0.05)
    conf, pred = probs.max(1), probs.argmax(1)
    rows = []
    for th in thresholds:
        m = conf >= th
        rows.append({"threshold": round(float(th), 2), "coverage": float(m.mean()),
                     "accuracy": float((pred[m] == y[m]).mean()) if m.any() else None})
    return rows


def pick_threshold(rows, min_coverage: float = 0.6, target_acc: float = 0.85) -> float:
    """Smallest threshold whose answered-accuracy >= target_acc while coverage >= min_coverage.
    Falls back to the best-accuracy row that still meets min_coverage. Fit on VALIDATION only."""
    ok = [r for r in rows if r["accuracy"] is not None and r["coverage"] >= min_coverage]
    good = [r for r in ok if r["accuracy"] >= target_acc]
    if good:
        return good[0]["threshold"]
    return max(ok, key=lambda r: r["accuracy"])["threshold"] if ok else 0.5
