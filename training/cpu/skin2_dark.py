"""Compute dark_spot_area for the val/test eval images (exact model input) -> RUN_DIR/dark_{split}.npy. Usage: python skin2_dark.py CACHE2 RUN_DIR"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
from skin_decide import dark_spot_area

c2, run = Path(sys.argv[1]), Path(sys.argv[2])
sp = pd.read_csv(run / "split.csv"); E = np.load(c2 / "eval224.npy", mmap_mode="r")
for s in ("val", "test"):
    ii = np.where(sp.split == s)[0]
    np.save(run / f"dark_{s}.npy", np.array([dark_spot_area(E[i]) for i in ii]))
    print(s, len(ii))
