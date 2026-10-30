"""Held-out TEST metrics, BEFORE (skin-v1 as shipped: squash of the whole photo, argmax-of-tier-sums, T=0.8767, thr 0.66) vs AFTER
(skin-v1.1: centre-square crop, healthy-skin class, sensitivity-first thresholds), overall and per source. Same test images for both.
Healthy-skin test patches are fed to the old model too (it has no such class: it must pick one of its 19).
Usage: python skin2_eval.py CACHE2 RUN_DIR EXPORT_DIR OLD_CACHE OLD_EXPORT OUT_METRICS_JSON REPO_RESULTS_JSON"""
import datetime, json, sys
from pathlib import Path
import numpy as np, onnxruntime as ort, pandas as pd
from scipy.special import softmax
from skin_classes import NAMES20 as NAMES
from skin_common import to_tensor
from skin_decide import DARK_MIN, TIER_IDX, decide, decide_old
from skin_metrics import bootstrap_generic, confusion, summarize

K = len(NAMES)
HEALTHY = NAMES.index("healthy-skin")
T_OLD, THR_OLD = 0.8767, 0.66


def metrics(P, code, y, mask):
    """P (n,K) probs, code (n,) decided tier 0..3, y (n,) class idx; subset mask."""
    P, code, y = P[mask], code[mask], y[mask]
    tt = TIER_IDX[y]
    red, nong, hs = tt == 2, tt >= 1, y == HEALTHY
    les = ~hs
    top3 = (np.argsort(-P, axis=1)[:, :3] == y[:, None]).any(1)
    nanm = lambda a, c: float(a[c].mean()) if c.any() else float("nan")
    ans = code != 3
    return dict(
        n=int(len(y)), n_red=int(red.sum()), n_healthy=int(hs.sum()),
        tier_accuracy=float((code == tt).mean()),                       # uncertain counts as wrong
        tier_accuracy_answered=nanm(code == tt, ans), coverage=float(ans.mean()),
        urgent_sensitivity=nanm(code == 2, red),                         # true red shown red
        urgent_specificity=nanm(code != 2, ~red),                        # not red shown not red
        red_or_amber_of_red=nanm(np.isin(code, (1, 2)), red),
        flagged_amber_or_red_of_nongreen=nanm(np.isin(code, (1, 2)), nong),
        healthy_false_alarm=nanm(np.isin(code, (1, 2)), hs),             # healthy patch shown amber/red
        healthy_not_green=nanm(code != 0, hs),                           # incl. uncertain
        healthy_top1=nanm(P.argmax(1) == HEALTHY, hs),
        top1=float((P.argmax(1) == y).mean()), top3=float(top3.mean()),
        top1_lesion19=nanm(P.argmax(1) == y, les), top3_lesion19=nanm(top3, les))


def clean(o):
    """NaN -> null (strict JSON for the dashboard)."""
    if isinstance(o, float) and o != o: return None
    if isinstance(o, dict): return {k: clean(v) for k, v in o.items()}
    if isinstance(o, list): return [clean(v) for v in o]
    return o


def main(c2, run, exp, old_cache, old_exp, out_json, repo_json):
    sp = pd.read_csv(run / "split.csv")
    te = np.where(sp.split == "test")[0]
    t = sp.iloc[te].reset_index(drop=True)
    y = t["cls"].map({n: i for i, n in enumerate(NAMES)}).to_numpy()
    tune = json.load(open(run / "tuning.json"))
    L_new = np.load(exp / "test_logits_int8.npy")
    P_new = softmax(L_new / tune["T"], axis=1)
    dark = np.load(run / "dark_test.npy")
    code_new = decide(P_new, tune["t_red"], tune["t_amber"], tune["thr"], True, tune["guard_min"], dark)
    code_nodark = decide(P_new, tune["t_red"], tune["t_amber"], tune["thr"], True, tune["guard_min"])
    # old model on the old pipeline (full photo squashed to 224) for lesion rows; healthy patches (square) through the same squash
    old_imgs = np.load(old_cache / "images_all.npy", mmap_mode="r")
    E = np.load(c2 / "eval224.npy", mmap_mode="r")
    s_old = ort.InferenceSession(str(old_exp / "skin-v1.int8.onnx"), providers=["CPUExecutionProvider"])
    L_old = np.zeros((len(te), 19), dtype=np.float32)
    for k, i in enumerate(te):
        r = int(sp.old_row[i])
        img = old_imgs[r] if r >= 0 else E[i]
        L_old[k] = s_old.run(None, {"input": to_tensor(img)[None]})[0][0]
    P_old = np.concatenate([softmax(L_old / T_OLD, axis=1), np.zeros((len(te), 1))], axis=1)
    code_old = decide_old(P_old, THR_OLD)
    srcs = {"overall": np.ones(len(t), bool), **{s: (t.source == s).to_numpy() for s in ("pad-ufes-20", "scin", "ham10000")}}
    res = {"after_without_dark_spot_rule": {k: metrics(P_new, code_nodark, y, m) for k, m in {"overall": np.ones(len(t), bool), "pad-ufes-20": (t.source == "pad-ufes-20").to_numpy(), "scin": (t.source == "scin").to_numpy()}.items()}, "before": {k: metrics(P_old, code_old, y, m) for k, m in srcs.items()}, "after": {k: metrics(P_new, code_new, y, m) for k, m in srcs.items()}}
    # cluster-bootstrap CIs (overall + phone sources) for the headline numbers
    grp = t.group.to_numpy()
    def ci(P, code, mask):
        idxs = np.where(mask)[0]
        def fn(ix):
            m = np.zeros(len(y), bool); m[idxs[ix]] = False
            sel = idxs[ix]
            mm = metrics(P[sel], code[sel], y[sel], np.ones(len(sel), bool))
            return {k: mm[k] for k in ("tier_accuracy", "urgent_sensitivity", "urgent_specificity", "healthy_false_alarm", "top1", "top3")}
        return bootstrap_generic(grp[idxs], fn, n_boot=400)
    res["ci95"] = {tag: {k: ci(P, c, m) for k, m in srcs.items() if k in ("overall", "pad-ufes-20", "scin")} for tag, P, c in (("before", P_old, code_old), ("after", P_new, code_new))}
    # 3-tier confusion (answered cases) + per-class (19+1) table for the new model
    ans = code_new != 3
    cm_tier = confusion(TIER_IDX[y][ans], code_new[ans], 3); st = summarize(cm_tier)
    cm_fine = confusion(y, P_new.argmax(1), K); sf = summarize(cm_fine)
    per_class = {n: dict(tier=["green", "amber", "red"][TIER_IDX[i]], support=int(cm_fine[i].sum()), sensitivity=float(sf["sens"][i]), specificity=float(sf["spec"][i]), f1=float(sf["f1"][i])) for i, n in enumerate(NAMES)}
    # by Fitzpatrick (SCIN dermatologist estimate) for the new model
    fst = {}
    for f in sorted(x for x in t.fst.dropna().unique() if x):
        mk = (t.fst == f).to_numpy(); mm = metrics(P_new, code_new, y, mk)
        fst[f] = dict(n=mm["n"], tier_accuracy=mm["tier_accuracy"], urgent_sensitivity=mm["urgent_sensitivity"], n_red=mm["n_red"], healthy_false_alarm=mm["healthy_false_alarm"], n_healthy=mm["n_healthy"])
    parity = json.load(open(exp / "parity.json"))
    now = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")
    full = dict(generatedAt=now, model="skin-v1.1.0", T=tune["T"], thresholds=dict(red=tune["t_red"], amber=tune["t_amber"], uncertain_green=tune["thr"], guard_min_melanoma=tune["guard_min"], dark_spot_min_area=DARK_MIN), tuning_val=tune["val"],
                n_test=int(len(te)), n_test_groups=int(t.group.nunique()), source_counts=t.source.value_counts().to_dict(), n_healthy_test=int((y == HEALTHY).sum()),
                results=res, per_class=per_class, tier_confusion_answered=cm_tier.tolist(), by_fitzpatrick_scin_derm_estimated=fst, parity=parity)
    json.dump(clean(full), open(repo_json, "w"), indent=1)
    a = res["after"]["overall"]
    dash = dict(
        schemaVersion=1, isSample=False, generatedAt=now, source="measured",
        description="MEASURED on a held-out, group-level (lesion/patient/case) test split of HAM10000 + PAD-UFES-20 + SCIN + healthy-skin patches cut from PAD-UFES-20 / SCIN photos. skin-v1.1.0 int8 ONNX (20 classes incl. healthy-skin), centre-square crop, sensitivity-first tier thresholds tuned on validation. 'before' = skin-v1.0.0 as shipped, same test images. Baseline for a school project: not a diagnostic device. Per-source before/after, 20-class results and Fitzpatrick breakdown are in the extra keys and in docs/model-cards/skin-v1.md.",
        dataset="HAM10000 (CC BY-NC 4.0) + PAD-UFES-20 (CC BY 4.0) + Google SCIN (SCIN Data Use License)", date=now[:10],
        splits={s: int((sp.split == s).sum()) for s in ("train", "val", "test")},
        models=dict(skin=dict(
            label="Skin image (skin-v1.1.0, measured)", classes=["green", "amber", "red"], accuracy=float(st["accuracy"]),
            sensitivity=float(st["sens"].mean()), specificity=float(st["spec"].mean()), f1=float(st["f1"].mean()),
            perClass={n: dict(sensitivity=float(st["sens"][i]), specificity=float(st["spec"][i])) for i, n in enumerate(["green", "amber", "red"])},
            confusionMatrix=cm_tier.tolist(), latencyMs=round(parity["int8_ms_per_img_desktop_cpu"], 1), sizeBytes=parity["size_int8"],
            testSetDescription=f"Held-out TEST split: {len(te)} images from {t.group.nunique()} lesions/patients/cases (HAM10000 {int((t.source=='ham10000').sum())}, PAD-UFES-20 {int((t.source=='pad-ufes-20').sum())}, SCIN {int((t.source=='scin').sum())}; {int((y==HEALTHY).sum())} are healthy-skin patches). Tier = sensitivity-first thresholds on class-probability sums; 'uncertain' answers are left out of the confusion matrix. Latency = desktop CPU onnxruntime, not a phone.")),
        fineGrained=dict(classes=NAMES, accuracy=a["top1"], top3_accuracy=a["top3"], perClass=per_class, confusionMatrix=cm_fine.tolist()),
        urgent=dict(sensitivity_all=a["urgent_sensitivity"], specificity=a["urgent_specificity"], red_or_amber_of_red=a["red_or_amber_of_red"], n_red=a["n_red"], thresholds=full["thresholds"]),
        uncertain=dict(coverage=a["coverage"], tier_accuracy_answered=a["tier_accuracy_answered"]),
        beforeAfter=res, ci95=res["ci95"], byFitzpatrickScinEstimated=fst)
    json.dump(clean(dash), open(out_json, "w"), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    a = [Path(x) for x in sys.argv[1:]]
    main(*a)
