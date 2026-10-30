"""Build the skin-v2 caches OUTSIDE the repo (CACHE2): per source, center-square eval images (224, exact app pipeline) + 256 px train
copies, and lesion-free healthy-skin patches (PAD-UFES-20 and SCIN).
Usage: python skin2_prep.py pad|ham|scin DATA_DIR CACHE2 [workers]"""
import io, subprocess, zlib, sys, zipfile
from multiprocessing import Pool
from pathlib import Path
import numpy as np, pandas as pd
from skin_pipe import center_square, eval224, healthy_patches, open_rgb, train256
from skin_classes import HAM, PAD, scin_class

N_SCIN = 26
PAD_PER_IMG, SCIN_PER_IMG = 2, 1


def one(im, rng, want_patches, per_img):
    sq = center_square(im)
    res = dict(e=eval224(sq), t=train256(sq), h=[])
    if want_patches:
        for p, st in healthy_patches(im, rng, per_img):
            res["h"].append((eval224(p), train256(p), st))
    return res


def pad_work(args):
    part, data, out = args
    dst = out / f"pad_{part}.npz"
    meta = pd.read_csv(data / "pad/metadata.csv").set_index("img_id")
    rows, E, T, hrows, HE, HT = [], [], [], [], [], []
    with zipfile.ZipFile(data / f"pad/imgs_part_{part}.zip") as zf:
        for n in sorted(zf.namelist()):
            if not n.lower().endswith(".png"):
                continue
            iid = Path(n).name
            r = meta.loc[iid]
            rng = np.random.default_rng(zlib.crc32(iid.encode()))
            res = one(open_rgb(io.BytesIO(zf.read(n))), rng, True, PAD_PER_IMG)
            E.append(res["e"]); T.append(res["t"])
            rows.append(dict(source="pad-ufes-20", image_id=iid, group="pad:" + r.patient_id, dx=r.diagnostic, cls=PAD[r.diagnostic], fst=""))
            for k, (e, t, st) in enumerate(res["h"]):
                HE.append(e); HT.append(t)
                hrows.append(dict(source="pad-ufes-20", image_id=f"{iid}#h{k}", group="pad:" + r.patient_id, dx="healthy", cls="healthy-skin", fst="", **st))
    np.savez(dst, E=np.stack(E), T=np.stack(T), HE=np.stack(HE), HT=np.stack(HT), rows=pd.DataFrame(rows).to_json(), hrows=pd.DataFrame(hrows).to_json())
    print("pad", part, len(rows), "healthy", len(hrows), flush=True)


def ham_work(args):
    part, data, out = args
    meta = pd.read_csv(data / "ham/meta.tab", sep="\t").set_index("image_id")
    rows, E, T = [], [], []
    with zipfile.ZipFile(data / f"ham/p{part}.zip") as zf:
        for n in sorted(zf.namelist()):
            if not n.lower().endswith(".jpg"):
                continue
            iid = Path(n).stem
            r = meta.loc[iid]
            res = one(open_rgb(io.BytesIO(zf.read(n))), None, False, 0)
            E.append(res["e"]); T.append(res["t"])
            rows.append(dict(source="ham10000", image_id=iid, group="ham:" + r.lesion_id, dx=r.dx, cls=HAM[r.dx], fst=""))
    np.savez(out / f"ham_{part}.npz", E=np.stack(E), T=np.stack(T), rows=pd.DataFrame(rows).to_json())
    print("ham", part, len(rows), flush=True)


SCIN_COLS = ["case_id", "weighted_skin_condition_label", "dermatologist_fitzpatrick_skin_type_label_1", "dermatologist_fitzpatrick_skin_type_label_2",
             "dermatologist_fitzpatrick_skin_type_label_3", "image_1_path", "image_2_path", "image_3_path", "image_1_shot_type", "image_2_shot_type", "image_3_shot_type"]


def fst_of(r):
    v = [r[f"dermatologist_fitzpatrick_skin_type_label_{k}"] for k in (1, 2, 3)]
    v = [x for x in v if isinstance(x, str) and x.startswith("FST")]
    return max(set(v), key=v.count) if v else ""


def scin_work(args):
    import pyarrow.parquet as pq
    i, data, out = args
    dst = out / f"scin_{i:02d}.npz"
    if dst.exists():
        return
    p = data / f"scin/s{i}.parquet"
    if not p.exists():
        url = f"https://huggingface.co/datasets/google/scin/resolve/main/data/train-{i:05d}-of-{N_SCIN:05d}.parquet"
        subprocess.run(["curl", "-sL", "--retry", "5", "-o", str(p), url], check=True)
    rows, E, T, hrows, HE, HT = [], [], [], [], [], []
    pf = pq.ParquetFile(p)
    for g in range(pf.metadata.num_row_groups):
        for r in pf.read_row_group(g, columns=SCIN_COLS).to_pylist():
            cls = scin_class(r["weighted_skin_condition_label"]) if r["weighted_skin_condition_label"] else None
            for k in (1, 2, 3):
                im = r[f"image_{k}_path"]
                if not (im and im["bytes"]):
                    continue
                try:
                    pil = open_rgb(io.BytesIO(im["bytes"]))
                except Exception:
                    continue
                rng = np.random.default_rng(int(abs(int(r["case_id"])) % (2 ** 32)) + k)
                res = one(pil, rng, True, SCIN_PER_IMG)
                iid = f"{r['case_id']}_{k}"
                if cls:
                    E.append(res["e"]); T.append(res["t"])
                    rows.append(dict(source="scin", image_id=iid, group="scin:" + r["case_id"], dx=cls, cls=cls, fst=fst_of(r)))
                for j, (e, t, st) in enumerate(res["h"]):
                    HE.append(e); HT.append(t)
                    hrows.append(dict(source="scin", image_id=f"{iid}#h{j}", group="scin:" + r["case_id"], dx="healthy", cls="healthy-skin", fst=fst_of(r),
                                      shot=r[f"image_{k}_shot_type"], **st))
    p.unlink()
    z = lambda a: np.stack(a) if a else np.zeros((0, 1, 1, 3), np.uint8)
    np.savez(dst, E=z(E), T=z(T), HE=z(HE), HT=z(HT), rows=pd.DataFrame(rows).to_json(), hrows=pd.DataFrame(hrows).to_json())
    print("scin", i, len(rows), "healthy", len(hrows), flush=True)


if __name__ == "__main__":
    what, data, out = sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3])
    nw = int(sys.argv[4]) if len(sys.argv) > 4 else 3
    out.mkdir(parents=True, exist_ok=True)
    if what == "pad":
        with Pool(min(nw, 3)) as pool: pool.map(pad_work, [(i, data, out) for i in (1, 2, 3)])
    elif what == "ham":
        with Pool(min(nw, 2)) as pool: pool.map(ham_work, [(i, data, out) for i in (1, 2)])
    else:
        with Pool(nw) as pool: list(pool.imap_unordered(scin_work, [(i, data, out) for i in range(N_SCIN)]))
    print("done", what)
