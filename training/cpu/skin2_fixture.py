"""Parity fixtures for skin-v1.1 -> src/inference/__fixtures__/skin (4 PNGs of different aspect ratios + expected logits + 300 decisions)
and e2e/fixtures. The raw datasets are not kept in the repo: fixture pixels are cached TEST images (224 px) resized (bicubic) to non-square
sizes, so the app's centre-square crop is exercised. Expected values come from Python (skin_pipe.eval224(center_square(open_rgb(png)))).
Usage: python skin2_fixture.py CACHE2 RUN_DIR EXPORT_DIR REPO_DIR"""
import json, shutil, sys
from pathlib import Path
import numpy as np, onnxruntime as ort, pandas as pd, torch
from PIL import Image
from scipy.special import softmax
from skin_classes import NAMES20 as NAMES
from skin_common import MEAN, STD, S, to_tensor
from skin_decide import dark_spot_area, decide_labels
from skin_export import load_net
from skin_pipe import center_square, eval224, open_rgb


def sharpness(im):
    a = np.asarray(im).astype(np.float64); g = 0.299 * a[..., 0] + 0.587 * a[..., 1] + 0.114 * a[..., 2]
    return float((g[1:-1, 2:] + g[1:-1, :-2] + g[2:, 1:-1] + g[:-2, 1:-1] - 4 * g[1:-1, 1:-1]).var())


def main(c2, run, exp, repo):
    sp = pd.read_csv(run / "split.csv")
    E = np.load(c2 / "eval224.npy", mmap_mode="r")
    tune = json.load(open(run / "tuning.json"))
    net = load_net(run)
    s8 = ort.InferenceSession(str(exp / "skin-v1.1.int8.onnx"), providers=["CPUExecutionProvider"])
    out = repo / "src/inference/__fixtures__/skin"; out.mkdir(parents=True, exist_ok=True)
    for f in out.glob("*"): f.unlink()
    items = []
    for name, src, cls, size in (("a", "ham10000", "melanoma", (384, 288)), ("b", "pad-ufes-20", "basal-cell-carcinoma", (320, 320)),
                                 ("c", "scin", "eczema", (270, 360)), ("d", "scin", "healthy-skin", (299, 224))):
        cand = np.where((sp.split == "test") & (sp.source == src) & (sp.cls == cls))[0]
        # the app refuses blurry photos (quality.ts, Laplacian variance < 20): pick the first candidate that survives the upscale
        for i in map(int, cand):
            Image.fromarray(np.asarray(E[i])).resize(size, Image.BICUBIC).save(out / f"{name}.png", optimize=True)
            if sharpness(center_square(open_rgb(out / f"{name}.png"))) >= 25: break
        u8 = eval224(center_square(open_rgb(out / f"{name}.png"))); u8.tofile(out / f"{name}.u8.bin")
        x = to_tensor(u8)[None]
        with torch.no_grad():
            l32 = net(torch.from_numpy(x)).numpy()[0]
        l8 = s8.run(None, {"input": x})[0][0]
        dk = dark_spot_area(u8)
        code, lab = decide_labels(softmax(l8[None] / tune["T"], axis=1), tune["t_red"], tune["t_amber"], tune["thr"], tune["guard_min"], np.array([dk]))
        items.append(dict(name=name, width=size[0], height=size[1], source=src, true_class=cls, logits_fp32=l32.round(5).tolist(), logits_int8=l8.round(5).tolist(),
                          dark_spot=round(dk, 5), tier=["green", "amber", "red", "uncertain"][code[0]], label=NAMES[lab[0]] if lab[0] >= 0 else "uncertain"))
        print(name, src, cls, "->", items[-1]["tier"], items[-1]["label"])
    L = np.load(exp / "test_logits_int8.npy")
    rng = np.random.default_rng(0)
    sel = rng.choice(len(L), 300, replace=False)
    dark = np.load(run / "dark_test.npy")
    code, lab = decide_labels(softmax(L[sel] / tune["T"], axis=1), tune["t_red"], tune["t_amber"], tune["thr"], tune["guard_min"], dark[sel])
    decisions = [dict(logits=L[j].round(4).tolist(), dark_spot=round(float(dark[j]), 5), tier=["green", "amber", "red", "uncertain"][c], label=NAMES[l] if l >= 0 else "uncertain") for j, c, l in zip(sel, code, lab)]
    json.dump(dict(size=S, mean=MEAN.tolist(), std=STD.tolist(), temperature=tune["T"], images=items, decisions=decisions), open(out / "expected.json", "w"))
    fx = repo / "e2e/fixtures"
    shutil.copyfile(out / "a.png", fx / "skin-real-a.png")
    for n in "bcd": shutil.copyfile(out / f"{n}.png", fx / f"skin-real-{n}.png")


if __name__ == "__main__":
    main(*(Path(a) for a in sys.argv[1:5]))
