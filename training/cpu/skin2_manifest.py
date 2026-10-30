"""Copy the skin-v1.1 int8 model into public/models (removing the v1.0 file) and rewrite the skin entry of the manifest.
Usage: python skin2_manifest.py RUN_DIR EXPORT_DIR REPO_DIR"""
import hashlib, json, shutil, sys
from pathlib import Path
from skin_classes import CLASSES20 as CLASSES, NAMES20 as NAMES


def main(run, exp, repo):
    tune = json.load(open(run / "tuning.json"))
    dst = repo / "public/models/skin-v1.1.int8.onnx"
    shutil.copyfile(exp / "skin-v1.1.int8.onnx", dst)
    old = repo / "public/models/skin-v1.int8.onnx"
    if old.exists(): old.unlink()
    data = dst.read_bytes()
    mpath = repo / "public/models/manifest.json"
    m = json.loads(mpath.read_text())
    entry = dict(
        id="skin", version="skin-v1.1.0", url="/models/skin-v1.1.int8.onnx", sha256=hashlib.sha256(data).hexdigest(), bytes=len(data),
        labels=NAMES, tiers={n: t for n, t in CLASSES},
        threshold=round(tune["thr"], 2), temperature=round(tune["T"], 4), tierRule="thresholds",
        tierThresholds=dict(red=round(tune["t_red"], 3), amber=round(tune["t_amber"], 3)),
        guard=dict(topIn=["nevus", "seborrheic-keratosis"], watch="melanoma", minWatchProb=round(tune["guard_min"], 3)),
        darkSpot=dict(minArea=tune["dark_min"], relLuma=0.45, maxLuma=70),
        onnx=dict(input="input", output="logits", imageSize=224, normalization="RGB, centre-square crop (camera: the guide box), shrink to <= 448 px, area-resize to 224x224, /255, ImageNet mean/std, NCHW float32"),
        notes="REAL baseline model (MobileNetV3-Large, fine-tuned, int8 per-channel weights). 20 classes: 19 skin conditions + healthy-skin (green). tierRule 'thresholds' "
              "(sensitivity-first): red if P(red) >= tierThresholds.red, else amber if P(amber)+P(red) >= tierThresholds.amber, else amber if the top class is nevus/seborrheic "
              "keratosis, melanoma is the runner-up and P(melanoma) >= guard.minWatchProb (guard), or an enclosed very dark spot covers >= darkSpot.minArea of the box (darkSpot), else uncertain if P(green) < threshold, else green. T, thresholds fitted on the validation split only. "
              "Input is the centre square (guide box) of the photo. Screening aid, not a diagnosis. Tiers of SCIN-derived classes are provisional (clinician review needed). See docs/model-cards/skin-v1.md.")
    m["models"] = [entry if e["id"] == "skin" else e for e in m["models"]]
    mpath.write_text(json.dumps(m, indent=2, ensure_ascii=False) + "\n")
    print(entry["sha256"], entry["bytes"], entry["threshold"], entry["temperature"], entry["tierThresholds"])


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]))
