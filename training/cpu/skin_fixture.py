"""Write 3 parity fixtures (one test image per source) into src/inference/__fixtures__/skin and e2e/fixtures.
The raw datasets are not kept, so the fixture pixels are the 224x224 cached TEST images bicubically upscaled to <= 480 px
(PNG, lossless): they exercise the area-resize path on a non-trivial size. Expected values come from Python.
Usage: python skin_fixture.py CACHE_DIR RUN_DIR EXPORT_DIR REPO_DIR"""
import json, sys
from pathlib import Path
import numpy as np, onnxruntime as ort, pandas as pd, torch
from PIL import Image
from skin_common import MEAN, STD, S, squash_u8, to_tensor
from skin_export import load_net


def main(cache, run, exp, repo):
    sp = pd.read_csv(run / "split.csv")
    imgs = np.load(cache / "images_all.npy", mmap_mode="r")
    net = load_net(run)
    s8 = ort.InferenceSession(str(exp / "skin-v1.int8.onnx"), providers=["CPUExecutionProvider"])
    out = repo / "src/inference/__fixtures__/skin"; out.mkdir(parents=True, exist_ok=True)
    items = []
    for name, src, cls, size in (("a", "ham10000", "melanoma", (480, 360)), ("b", "pad-ufes-20", "basal-cell-carcinoma", (400, 400)), ("c", "scin", "eczema", (360, 480))):
        i = int(np.where((sp.split == "test") & (sp.source == src) & (sp.cls == cls))[0][0])
        im = Image.fromarray(np.asarray(imgs[i])).resize(size, Image.BICUBIC)
        im.save(out / f"{name}.png", optimize=True)
        rgb = np.asarray(Image.open(out / f"{name}.png").convert("RGB"))
        u8 = squash_u8(rgb); u8.tofile(out / f"{name}.u8.bin")
        x = to_tensor(u8)[None]
        with torch.no_grad():
            l32 = net(torch.from_numpy(x)).numpy()[0]
        l8 = s8.run(None, {"input": x})[0][0]
        items.append(dict(name=name, width=size[0], height=size[1], source=src, true_class=cls, logits_fp32=l32.round(5).tolist(), logits_int8=l8.round(5).tolist()))
        print(name, src, cls, "top fp32", int(l32.argmax()), "top int8", int(l8.argmax()))
    json.dump(dict(size=S, mean=MEAN.tolist(), std=STD.tolist(), images=items), open(out / "expected.json", "w"))
    import shutil; shutil.copyfile(out / "a.png", repo / "e2e/fixtures/skin-real-a.png")


if __name__ == "__main__":
    main(*(Path(a) for a in sys.argv[1:5]))
