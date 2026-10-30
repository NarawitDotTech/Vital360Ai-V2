"""Held-out TEST metrics of the shipped int8 model at two levels: 19 fine classes and the derived 3 tiers
(tier = argmax of summed class probabilities; 'uncertain' below threshold). Cluster-bootstrap 95% CIs over patients/lesions/cases.
Usage: python skin_eval.py CACHE_DIR RUN_DIR EXPORT_DIR OUT_METRICS_JSON"""
import json, sys, datetime
from pathlib import Path
import numpy as np, pandas as pd
from scipy.special import softmax
from skin_calib import TIER_IDX, TIER_MAT
from skin_classes import NAMES, TIER_NAMES
from skin_metrics import bootstrap_generic, confusion, summarize

K = len(NAMES)


def per_class_table(cm, names):
    s = summarize(cm)
    return {n: dict(support=int(cm[i].sum()), sensitivity=float(s["sens"][i]), specificity=float(s["spec"][i]), f1=float(s["f1"][i])) for i, n in enumerate(names)}


def main(cache, run, exp, out_json):
    sp = pd.read_csv(run / "split.csv")
    h = np.load(run / "head.npz"); T, thr = float(h["T"]), float(h["thr"])
    te = np.where(sp.split == "test")[0]
    t = sp.iloc[te].reset_index(drop=True)
    y = t["cls"].map({n: i for i, n in enumerate(NAMES)}).to_numpy()
    lg = np.load(exp / "test_logits_int8.npy")
    lg_ref = np.load(exp / "test_logits_torch.npy")
    parity = json.load(open(exp / "parity.json"))
    res = {}
    for tag, L in (("int8", lg), ("fp32", lg_ref)):
        p = softmax(L / T, axis=1)
        fine = p.argmax(1)
        tp = p @ TIER_MAT
        tier_pred = tp.argmax(1); conf = tp.max(1)
        tier_true = TIER_IDX[y]
        res[tag] = dict(p=p, fine=fine, tier_pred=tier_pred, conf=conf, tier_true=tier_true)
    r = res["int8"]
    cm_fine = confusion(y, r["fine"], K)
    cm_tier = confusion(r["tier_true"], r["tier_pred"], 3)
    sf, st = summarize(cm_fine), summarize(cm_tier)
    answered = r["conf"] >= thr
    red = r["tier_true"] == 2
    top3 = (np.argsort(-r["p"], axis=1)[:, :3] == y[:, None]).any(1).mean()
    def tier_stats(mask):
        m = confusion(r["tier_true"][mask], r["tier_pred"][mask], 3); s = summarize(m)
        return dict(n=int(mask.sum()), accuracy=float(s["accuracy"]), sens_red=float(s["sens"][2]) if m[2].sum() else None, n_red=int(m[2].sum()))
    breakdown = {src: tier_stats((t.source == src).to_numpy()) | dict(fine_accuracy=float((r["fine"][(t.source == src).to_numpy()] == y[(t.source == src).to_numpy()]).mean())) for src in t.source.unique()}
    fst = {}
    for f in sorted(x for x in t.fst.dropna().unique() if x):
        mk = (t.fst == f).to_numpy()
        fst[f] = tier_stats(mk)
    # PAD Fitzpatrick from original metadata
    pad = pd.read_csv("/home/user/ml/data/pad/metadata.csv").set_index("img_id")
    padm = (t.source == "pad-ufes-20").to_numpy()
    padf = t[padm].image_id.map(pad["fitspatrick"])
    fst_pad = {f"FST{int(f)}": tier_stats((padm & (t.image_id.map(pad["fitspatrick"]) == f).to_numpy())) for f in sorted(padf.dropna().unique())}
    red_flag = float(((r["tier_pred"] == 2) | ~answered)[red].mean())  # red OR uncertain -> user is told to see someone
    urgent = dict(sensitivity_all=float(st["sens"][2]), sensitivity_answered=float((r["tier_pred"][red & answered] == 2).mean()) if (red & answered).any() else None,
                  flagged_red_or_uncertain=red_flag, n_red=int(red.sum()))
    cis = bootstrap_generic(t.group.to_numpy(), lambda idx: dict(
        fine_accuracy=float((r["fine"][idx] == y[idx]).mean()), tier_accuracy=float((r["tier_pred"][idx] == r["tier_true"][idx]).mean()),
        urgent_sensitivity=float((r["tier_pred"][idx][r["tier_true"][idx] == 2] == 2).mean()) if (r["tier_true"][idx] == 2).any() else np.nan,
        tier_macro_f1=float(summarize(confusion(r["tier_true"][idx], r["tier_pred"][idx], 3))["f1"].mean())))
    full = dict(
        generatedAt=datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"), T=T, threshold=thr, n_test=int(len(te)),
        n_test_groups=int(t.group.nunique()), source_counts=t.source.value_counts().to_dict(),
        fine=dict(accuracy=float(sf["accuracy"]), macro_sens=float(sf["sens"][np.bincount(y, minlength=K) > 0].mean()), macro_f1=float(sf["f1"][np.bincount(y, minlength=K) > 0].mean()), top3_accuracy=float(top3),
                  perClass=per_class_table(cm_fine, NAMES), confusionMatrix=cm_fine.tolist()),
        tier=dict(accuracy=float(st["accuracy"]), perClass=per_class_table(cm_tier, TIER_NAMES), confusionMatrix=cm_tier.tolist()),
        uncertain=dict(coverage=float(answered.mean()), tier_accuracy_answered=float((r["tier_pred"][answered] == r["tier_true"][answered]).mean())),
        urgent=urgent, by_source=breakdown, by_fitzpatrick_scin_derm_estimated=fst, by_fitzpatrick_pad=fst_pad, ci95=cis,
        fp32_tier_accuracy=float((res["fp32"]["tier_pred"] == res["fp32"]["tier_true"]).mean()), fp32_fine_accuracy=float((res["fp32"]["fine"] == y).mean()), parity=parity)
    json.dump(full, open(run / "eval_full.json", "w"), indent=1)
    dash = dict(
        schemaVersion=1, isSample=False, generatedAt=full["generatedAt"], source="measured",
        description="MEASURED on a held-out, group-level (lesion/patient/case) test split of HAM10000 + PAD-UFES-20 + SCIN. skin-v1 int8 ONNX, tiers derived from 19 fine classes. Baseline for a school project: not a diagnostic device. Fine-grained 19-class results, confidence intervals and per-source / per-skin-tone breakdowns are in the extra keys below and in docs/model-cards/skin-v1.md.",
        dataset="HAM10000 (CC BY-NC 4.0) + PAD-UFES-20 (CC BY 4.0) + Google SCIN (SCIN Data Use License)", date=full["generatedAt"][:10],
        splits={s: int((sp.split == s).sum()) for s in ("train", "val", "test")},
        models=dict(skin=dict(
            label="Skin image (skin-v1, measured)", classes=TIER_NAMES, accuracy=full["tier"]["accuracy"],
            sensitivity=float(st["sens"].mean()), specificity=float(st["spec"].mean()), f1=float(st["f1"].mean()),
            perClass={n: dict(sensitivity=v["sensitivity"], specificity=v["specificity"]) for n, v in full["tier"]["perClass"].items()},
            confusionMatrix=cm_tier.tolist(), latencyMs=round(parity["int8_ms_per_img_desktop_cpu"], 1), sizeBytes=parity["size_int8"],
            testSetDescription=f"Held-out TEST split: {len(te)} images from {t.group.nunique()} lesions/patients/cases (HAM10000 {int((t.source=='ham10000').sum())}, PAD-UFES-20 {int((t.source=='pad-ufes-20').sum())}, SCIN {int((t.source=='scin').sum())}). Tiers green/amber/red derived from the 19-class output by summing probabilities. Latency = desktop CPU onnxruntime, not a phone.")),
        fineGrained=full["fine"] | dict(classes=NAMES), urgent=urgent, uncertain=full["uncertain"], bySource=breakdown,
        byFitzpatrickScinEstimated=fst, byFitzpatrickPad=fst_pad, ci95=cis)
    json.dump(dash, open(out_json, "w"), indent=1)
    print(json.dumps({k: full[k] for k in ("n_test", "fine", "tier", "uncertain", "urgent", "by_source", "ci95")}, indent=1, default=str)[:6000])


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]), Path(sys.argv[4]))
