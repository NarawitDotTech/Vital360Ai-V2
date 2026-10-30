"""Temperature scaling and uncertainty threshold for the fine-grained head with tier-sum decisions."""
import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import log_softmax, softmax
from skin_classes import CLASSES, TIER_NAMES

TIER_IDX = np.array([TIER_NAMES.index(t) for _, t in CLASSES])
TIER_MAT = np.eye(3)[TIER_IDX]  # (K, 3) class -> tier one-hot


def nll(logits, y, T):
    return -log_softmax(logits / T, axis=1)[np.arange(len(y)), y].mean()


def fit_temperature(logits, y):
    return float(minimize_scalar(lambda t: nll(logits, y, t), bounds=(0.2, 10), method="bounded").x)


def tier_probs(logits, T):
    return softmax(logits / T, axis=1) @ TIER_MAT


def pick_threshold(tp, tier_true, min_coverage=0.6, gain=0.05):
    """Lowest threshold on the winning tier prob at which accuracy on the confident subset beats the all-cases tier accuracy
    by `gain`, while still answering for >= min_coverage of val cases (else 0.5)."""
    pred = tp.argmax(1)
    base = float((pred == tier_true).mean())
    for t in np.arange(0.34, 0.99, 0.01):
        m = tp.max(1) >= t
        if m.mean() < min_coverage:
            break
        if (pred[m] == tier_true[m]).mean() >= base + gain:
            return float(round(t, 2)), base
    return 0.5, base
