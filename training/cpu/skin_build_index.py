"""Merge HAM10000 + PAD-UFES-20 (cache/index.csv, images224.npy) and SCIN shards into one index + one array.
Usage: python skin_build_index.py CACHE_DIR   -> cache/index_all.csv, cache/images_all.npy"""
import glob, io, sys
from pathlib import Path
import numpy as np, pandas as pd
from skin_classes import HAM, PAD, scin_class


def fst(r):
    v = [x for x in (r.efst1, r.efst2, r.efst3) if isinstance(x, str) and x.startswith("FST")]
    return max(set(v), key=v.count) if v else ""


def main(cache: Path):
    old = pd.read_csv(cache / "index.csv")
    imgs = [np.load(cache / "images224.npy", mmap_mode="r")]
    old["cls"] = [HAM[d] if s == "ham10000" else PAD[d] for s, d in zip(old.source, old.dx)]
    old["fst"] = ""
    rows = [old[["source", "image_id", "group", "dx", "cls", "fst"]]]
    for f in sorted(glob.glob(str(cache / "scin/scin_*.npz"))):
        z = np.load(f, allow_pickle=True)
        meta = pd.read_json(io.StringIO(str(z["meta"])), dtype={"case_id": str})
        keep = [i for i, w in enumerate(meta.wlabel) if scin_class(w)]
        if not keep:
            continue
        m = meta.iloc[keep]
        imgs.append(z["imgs"][keep])
        rows.append(pd.DataFrame(dict(source="scin", image_id=m.case_id + "_" + m.k.astype(str), group="scin:" + m.case_id.astype(str),
                                      dx=[scin_class(w) for w in m.wlabel], cls=[scin_class(w) for w in m.wlabel], fst=[fst(r) for r in m.itertuples()])))
    idx = pd.concat(rows, ignore_index=True)
    np.save(cache / "images_all.npy", np.concatenate(imgs))
    idx.to_csv(cache / "index_all.csv", index=False)
    print(idx.groupby(["cls", "source"]).size().unstack(fill_value=0))
    print(idx.groupby("cls").group.nunique().sort_values())


if __name__ == "__main__":
    main(Path(sys.argv[1]))
