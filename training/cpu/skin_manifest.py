"""Copy the int8 model into public/models and update the skin entry of the manifest.
Usage: python skin_manifest.py RUN_DIR EXPORT_DIR REPO_DIR"""
import hashlib, json, shutil, sys
from pathlib import Path
import numpy as np
from skin_classes import CLASSES, NAMES


def main(run, exp, repo):
    h = np.load(run / "head.npz")
    dst = repo / "public/models/skin-v1.int8.onnx"
    shutil.copyfile(exp / "skin-v1.int8.onnx", dst)
    data = dst.read_bytes()
    mpath = repo / "public/models/manifest.json"
    m = json.loads(mpath.read_text())
    entry = dict(
        id="skin", version="skin-v1.0.0", url="/models/skin-v1.int8.onnx", sha256=hashlib.sha256(data).hexdigest(), bytes=len(data),
        labels=NAMES, tiers={n: t for n, t in CLASSES},
        threshold=round(float(h["thr"]), 2), temperature=round(float(h["T"]), 4), tierRule="sum",
        onnx=dict(input="input", output="logits", imageSize=224, normalization="RGB, area-resize squash to 224x224, /255, ImageNet mean/std, NCHW float32"),
        notes="REAL baseline model (MobileNetV3-Large, fine-tuned, int8 static QDQ). 19 classes, tier = argmax of summed class probabilities (tierRule sum), "
              "uncertain when that tier probability < threshold. Temperature and threshold fitted on the validation split only. Screening aid, not a diagnosis. "
              "Tier assignments of the SCIN-derived classes are provisional and need clinician review. See docs/model-cards/skin-v1.md.")
    m["models"] = [entry if e["id"] == "skin" else e for e in m["models"]]
    mpath.write_text(json.dumps(m, indent=2, ensure_ascii=False) + "\n")
    print(entry["sha256"], entry["bytes"], entry["threshold"], entry["temperature"])


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]))
