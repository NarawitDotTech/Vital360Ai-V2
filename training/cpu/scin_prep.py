"""Download Google SCIN (HF google/scin, SCIN Data Use License) shard by shard, keep a 224x224 cache of every image
(+ label metadata), delete each shard after use. Usage: python scin_prep.py DATA_DIR CACHE_DIR [workers]"""
import io, subprocess, sys
from multiprocessing import Pool
from pathlib import Path
import numpy as np, pandas as pd, pyarrow.parquet as pq
from skin_common import load_rgb, squash_u8

N = 26
COLS = ["case_id", "weighted_skin_condition_label", "fitzpatrick_skin_type", "dermatologist_fitzpatrick_skin_type_label_1",
        "dermatologist_fitzpatrick_skin_type_label_2", "dermatologist_fitzpatrick_skin_type_label_3",
        "image_1_path", "image_2_path", "image_3_path"]


def work(args):
    i, data, cache = args
    out = cache / f"scin_{i:02d}.npz"
    if out.exists():
        return i
    p = data / f"s{i}.parquet"
    url = f"https://huggingface.co/datasets/google/scin/resolve/main/data/train-{i:05d}-of-{N:05d}.parquet"
    subprocess.run(["curl", "-sL", "--retry", "5", "-o", str(p), url], check=True)
    meta, arrs = [], []
    pf = pq.ParquetFile(p)
    for g in range(pf.metadata.num_row_groups):
        for r in pf.read_row_group(g, columns=COLS).to_pylist():
            for k in (1, 2, 3):
                im = r[f"image_{k}_path"]
                if im and im["bytes"]:
                    arrs.append(squash_u8(load_rgb(io.BytesIO(im["bytes"]))))
                    meta.append(dict(case_id=r["case_id"], k=k, wlabel=r["weighted_skin_condition_label"], sfst=r["fitzpatrick_skin_type"],
                                     efst1=r["dermatologist_fitzpatrick_skin_type_label_1"], efst2=r["dermatologist_fitzpatrick_skin_type_label_2"],
                                     efst3=r["dermatologist_fitzpatrick_skin_type_label_3"]))
    p.unlink()
    np.savez(out, imgs=np.stack(arrs), meta=pd.DataFrame(meta).to_json())
    print("shard", i, len(arrs), flush=True)
    return i


if __name__ == "__main__":
    data, cache = Path(sys.argv[1]), Path(sys.argv[2])
    cache.mkdir(parents=True, exist_ok=True)
    with Pool(int(sys.argv[3]) if len(sys.argv) > 3 else 3) as pool:
        list(pool.imap_unordered(work, [(i, data, cache) for i in range(N)]))
    print("done")
