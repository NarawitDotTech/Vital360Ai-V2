"""Contact sheet of random healthy-skin patches (rows: PAD train, SCIN train, PAD test, SCIN test) for the visual spot check.
Usage: python skin2_sheet.py CACHE2 OUT.jpg"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
from PIL import Image, ImageDraw


def main(c2, out):
    idx = pd.read_csv(c2 / "index2.csv")
    E = np.load(c2 / "eval224.npy", mmap_mode="r"); T = np.load(c2 / "train256.npy", mmap_mode="r")
    rng = np.random.default_rng(3)
    sz, cols = 112, 16
    rows = []
    for split, src in (("train", "pad-ufes-20"), ("train", "scin"), ("test", "pad-ufes-20"), ("test", "scin")):
        ii = np.where((idx.cls == "healthy-skin") & (idx.split == split) & (idx.source == src))[0]
        pick = rng.choice(ii, 32, replace=False)
        rows.append([(Image.fromarray(np.asarray(T[i] if split == "train" else E[i]))).resize((sz, sz)) for i in pick])
    W = Image.new("RGB", (cols * sz, 8 * sz), "white")
    for r, ims in enumerate(rows):
        for k, im in enumerate(ims): W.paste(im, ((k % cols) * sz, (r * 2 + k // cols) * sz))
    W.save(out, quality=82)


if __name__ == "__main__":
    main(Path(sys.argv[1]), sys.argv[2])
