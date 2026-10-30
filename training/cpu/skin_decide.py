"""skin-v1.1 tier decision on class probabilities. MUST match src/inference/calibrate.ts finalizeByThresholds (tierRule "thresholds").
Codes: 0 green, 1 amber, 2 red, 3 uncertain."""
import numpy as np
from scipy.ndimage import label
from skin_classes import CLASSES20 as CLASSES, NAMES20 as NAMES

TIER_IDX = np.array([["green", "amber", "red"].index(t) for _, t in CLASSES])
TIER_MAT = np.eye(3)[TIER_IDX]
NEV_LIKE = [NAMES.index("nevus"), NAMES.index("seborrheic-keratosis")]
MEL = NAMES.index("melanoma")


DARK_REL, DARK_ABS, DARK_MIN = 0.45, 70.0, 0.03  # manifest.darkSpot {relLuma, maxLuma, minArea}; MUST match src/inference/skin-preprocess.ts darkSpotArea


def dark_spot_area(u8):
    """u8: 224x224x3 uint8 (the exact model input image). Area (fraction of the box) of the largest very dark blob that does NOT touch the
    border (an enclosed black spot, not a shadow / vignette / dark background): luma on a 56x56 grid (4x4 mean), dark = luma < 0.45*median
    and < 70, 4-connected components."""
    a = np.asarray(u8).astype(np.float64)
    Y = (0.299 * a[..., 0] + 0.587 * a[..., 1] + 0.114 * a[..., 2]).reshape(56, 4, 56, 4).mean((1, 3))
    dark = (Y < DARK_REL * np.median(Y)) & (Y < DARK_ABS)
    lab, n = label(dark)
    best = 0
    for k in range(1, n + 1):
        c = lab == k
        if c[0].any() or c[-1].any() or c[:, 0].any() or c[:, -1].any():
            continue
        best = max(best, int(c.sum()))
    return best / (56 * 56)


def tier_sums(P):
    return P @ TIER_MAT


def guard_hit(P, min_watch=0.0):
    """top class nevus/seborrheic-keratosis and melanoma is the runner-up (with P(melanoma) >= min_watch; 0 = literal rule)."""
    order = np.argsort(-P, axis=1)
    return np.isin(order[:, 0], NEV_LIKE) & (order[:, 1] == MEL) & (P[:, MEL] >= min_watch)


def decide(P, t_red, t_amber, thr, guard=True, guard_min=0.0, dark=None, dark_min=DARK_MIN):
    tp = tier_sums(P)
    out = np.zeros(len(P), dtype=int)  # green
    green_unsure = tp[:, 0] < thr
    out[green_unsure] = 3
    if guard:
        out[guard_hit(P, guard_min)] = 1
    if dark is not None:
        out[np.asarray(dark) >= dark_min] = 1
    out[tp[:, 1] + tp[:, 2] >= t_amber] = 1
    out[tp[:, 2] >= t_red] = 2
    return out


def decide_old(P, thr):
    """skin-v1 rule (tierRule "sum"): argmax of tier sums, uncertain below thr."""
    tp = tier_sums(P)
    out = tp.argmax(1)
    out[tp.max(1) < thr] = 3
    return out


def decide_labels(P, t_red, t_amber, thr, guard_min=0.0, dark=None, dark_min=DARK_MIN):
    """Same branches as calibrate.ts finalizeByThresholds: returns (tier code 0..3, label index or -1 for uncertain)."""
    tp = tier_sums(P)
    top = P.argmax(1)
    gh = guard_hit(P, guard_min)
    code = np.zeros(len(P), dtype=int); force = np.zeros(len(P), bool)
    for i in range(len(P)):
        if tp[i, 2] >= t_red: code[i] = 2
        elif tp[i, 1] + tp[i, 2] >= t_amber: code[i] = 1
        elif gh[i] or (dark is not None and dark[i] >= dark_min): code[i], force[i] = 1, True
        elif tp[i, 0] < thr: code[i] = 3
    lab = top.copy()
    for i in range(len(P)):
        if code[i] == 3: lab[i] = -1
        elif not force[i] and TIER_IDX[top[i]] != code[i]:
            m = np.where(TIER_IDX == code[i])[0]
            lab[i] = m[np.argmax(P[i, m])]
    return code, lab
