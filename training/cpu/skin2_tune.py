"""Tune the skin-v1.1 decision thresholds on VALIDATION only (int8 logits of the shipped model).
  T      : temperature, minimises NLL on val
  t_red  : largest P(red) cut-off that still shows >= 0.90 of val urgent-referral (red) cases as red
  guard_min: largest P(melanoma) floor for the nevus/SK -> amber guard that keeps the guard's full val gain on red cases (floor 0 = literal rule
             "melanoma is runner-up"; the literal rule flags ~60 % of nevi amber, a 5 % floor keeps the same red-case gain at about a third of the false alarms)
  t_amber: largest P(amber)+P(red) cut-off (>= t_red) that still shows (amber or red) >= 0.80 of val amber-tier cases
  thr    : uncertain cut-off on P(green): lowest value where answered cases are >= 5 points more accurate and coverage stays >= 0.6
Usage: python skin2_tune.py RUN_DIR EXPORT_DIR"""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
from scipy.special import softmax
from skin_calib import fit_temperature
from skin_classes import NAMES20 as NAMES
from skin_decide import DARK_MIN, TIER_IDX, decide, guard_hit, tier_sums, MEL

TARGET_RED, TARGET_AMBER = 0.90, 0.80


def main(run: Path, exp: Path):
    sp = pd.read_csv(run / "split.csv")
    va = np.where(sp.split == "val")[0]
    y = sp["cls"].map({n: i for i, n in enumerate(NAMES)}).to_numpy()[va]
    tt = TIER_IDX[y]
    L = np.load(exp / "val_logits_int8.npy")
    dark = np.load(run / "dark_val.npy")  # skin2_dark.py: enclosed dark-spot area of each val image (fixed rule, not tuned: DARK_MIN)
    T = fit_temperature(L, y)
    P = softmax(L / T, axis=1)
    tp = tier_sums(P)
    red = tt == 2
    # t_red
    grid = np.round(np.arange(0.01, 0.99, 0.005), 3)
    sens = np.array([(tp[red, 2] >= t).mean() for t in grid])
    ok = grid[sens >= TARGET_RED]
    t_red = float(ok.max())
    # guard floor: extra amber alerts on red-truth cases that the thresholds alone leave green/uncertain
    base0 = decide(P, t_red, 1.01, 0.0, guard=False)
    gain = lambda f: float((guard_hit(P, f) & np.isin(base0, (0, 3)))[red].mean())
    guard_min = float(max(f for f in np.round(np.arange(0, 0.201, 0.01), 2) if gain(f) >= gain(0.0) - 1e-9))
    # t_amber (amber or red shown), given t_red and the guard
    nong = tt >= 1
    amb = tt == 1
    flag = lambda t: ((tp[:, 1] + tp[:, 2] >= t) | (tp[:, 2] >= t_red) | guard_hit(P, guard_min) | (dark >= DARK_MIN))[amb].mean()
    ga = np.round(np.arange(t_red, 0.99, 0.005), 3)
    okA = [t for t in ga if flag(t) >= TARGET_AMBER]
    t_amber = float(max(okA)) if okA else float(t_red)
    # uncertain threshold on P(green) for would-be-green cases
    base_pred = decide(P, t_red, t_amber, 0.0, True, guard_min, dark)
    base = float((base_pred == tt).mean())
    thr = 0.66  # fallback = skin-v1 value. A green answer already has P(green) >= 1 - t_amber, so values <= that are inert.
    for c in np.round(np.arange(max(0.5, 1 - t_amber + 0.01), 0.99, 0.01), 2):
        d = decide(P, t_red, t_amber, c, True, guard_min, dark)
        m = d != 3
        if m.mean() < 0.6: break
        if (d[m] == tt[m]).mean() >= base + 0.05:
            thr = float(c); break
    d = decide(P, t_red, t_amber, thr, True, guard_min, dark)
    hs = y == NAMES.index("healthy-skin")
    rep = dict(T=T, t_red=t_red, t_amber=t_amber, thr=thr, guard_min=guard_min, dark_min=DARK_MIN, guard_gain_red_truth=gain(guard_min), guard_gain_literal=gain(0.0), val=dict(
        n=int(len(va)), n_red=int(red.sum()), sens_red_shown=float((d[red] == 2).mean()), spec_red=float((d[~red] != 2).mean()),
        flag_nongreen=float((d[nong] >= 1).mean() if True else 0), flag_nongreen_amber_or_red=float(np.isin(d[nong], (1, 2)).mean()),
        green_truth_flagged=float(np.isin(d[tt == 0], (1, 2)).mean()), healthy_n=int(hs.sum()), healthy_false_alarm=float(np.isin(d[hs], (1, 2)).mean()),
        healthy_not_green=float((d[hs] != 0).mean()), tier_acc_strict=float((d == tt).mean()), coverage=float((d != 3).mean()),
        tier_acc_answered=float((d[d != 3] == tt[d != 3]).mean())),
        curve_t_red=[dict(t=float(t), sens_red=float((tp[red, 2] >= t).mean()), spec_red=float((tp[~red, 2] < t).mean())) for t in (0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5)],
        targets=dict(sens_red=TARGET_RED, amber_truth_flagged=TARGET_AMBER))
    json.dump(rep, open(run / "tuning.json", "w"), indent=1)
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
