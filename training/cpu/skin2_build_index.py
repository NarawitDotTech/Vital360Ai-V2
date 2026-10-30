"""Merge the skin2 caches into one eval array (224, exact app pipeline), one train array (256) and one index with a leak-free split.
Split: lesion images keep their skin-v1 split (group-level, seed 42, 7-fold stratified; fold0 test, fold1 val); healthy patches inherit the
split of their source group (PAD patient / SCIN case). Groups never seen before (SCIN cases that had no class label) are assigned by a hash
of the case id (1/7 test, 1/7 val). Asserts no group is in two splits.
Disk saving: eval224.npy is only filled for val/test rows and train256.npy only for train rows (the files are sparse; other rows stay 0).
Usage: python skin2_build_index.py CACHE2 OLD_RUN_DIR  (OLD_RUN_DIR/split.csv = skin-v1 split) -> CACHE2/{eval224.npy,train256.npy,index2.csv}"""
import io, sys, zlib
from pathlib import Path
import numpy as np, pandas as pd

SCIN_HEALTHY_CAP = 3200


def rj(z, k):
    return pd.read_json(io.StringIO(str(z[k])), dtype={"image_id": str, "group": str})


def main(c2: Path, old_run: Path):
    parts, rows = [], []  # pass 1: metadata only (npz arrays load lazily)
    for f in ["pad_1", "pad_2", "pad_3", "ham_1", "ham_2"] + [f"scin_{i:02d}" for i in range(26)]:
        p = c2 / f"{f}.npz"
        if not p.exists():
            print("MISSING", f); continue
        z = np.load(p, allow_pickle=True)
        for ek, tk, rk in (("E", "T", "rows"), ("HE", "HT", "hrows")):
            if rk in z.files:
                r = rj(z, rk)
                if len(r):
                    rows.append(r); parts.append((p, ek, tk, len(r)))
    idx = pd.concat(rows, ignore_index=True)
    rng = np.random.default_rng(0)  # cap SCIN healthy patches (random subset), keep all PAD healthy
    scin_h = np.where((idx.source == "scin") & (idx.cls == "healthy-skin"))[0]
    drop = set(rng.permutation(scin_h)[SCIN_HEALTHY_CAP:].tolist()) if len(scin_h) > SCIN_HEALTHY_CAP else set()
    keep = np.array([i for i in range(len(idx)) if i not in drop])
    new_pos = {int(o): n for n, o in enumerate(keep)}
    idx = idx.iloc[keep].reset_index(drop=True)
    old = pd.read_csv(old_run / "split.csv")
    old["old_row"] = np.arange(len(old))
    key = old.set_index(["source", "image_id"])
    gmap = old.groupby("group").split.first().to_dict()
    sp, orow = [], []
    for r in idx.itertuples():
        if r.cls != "healthy-skin" and (r.source, r.image_id) in key.index:
            o = key.loc[(r.source, r.image_id)]
            sp.append(o.split); orow.append(int(o.old_row))
        else:
            orow.append(-1)
            if r.group in gmap:
                sp.append(gmap[r.group])
            else:
                h = zlib.crc32(r.group.encode()) % 7
                sp.append("test" if h == 0 else "val" if h == 1 else "train")
    idx["split"], idx["old_row"] = sp, orow
    g = idx.groupby("group").split.nunique()
    assert (g == 1).all(), g[g > 1].head()
    print("lesion rows without old split:", int(((idx.cls != "healthy-skin") & (idx.old_row < 0)).sum()))
    is_train = (idx.split == "train").to_numpy()
    E = np.lib.format.open_memmap(c2 / "eval224.npy", mode="w+", dtype=np.uint8, shape=(len(idx), 224, 224, 3))
    T = np.lib.format.open_memmap(c2 / "train256.npy", mode="w+", dtype=np.uint8, shape=(len(idx), 256, 256, 3))
    off = 0  # pass 2: stream the arrays, deleting each part file once consumed (parts that hold both E/HE are deleted after their last use)
    for k, (p, ek, tk, n) in enumerate(parts):
        z = np.load(p, allow_pickle=True)
        e, t = z[ek], z[tk]
        for j in range(n):
            q = new_pos.get(off + j)
            if q is None: continue
            if is_train[q]: T[q] = t[j]
            else: E[q] = e[j]
        off += n
        del e, t, z
        if k + 1 == len(parts) or parts[k + 1][0] != p:
            p.unlink()
    E.flush(); T.flush()
    idx.to_csv(c2 / "index2.csv", index=False)
    print(len(idx)); print(idx.groupby(["cls", "split"]).size().unstack(fill_value=0)); print(idx.groupby(["source", "split"]).size().unstack(fill_value=0))


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
