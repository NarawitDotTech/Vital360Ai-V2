"""Build the skin training cache OUTSIDE the repo: 224x224 uint8 arrays + an index CSV.

Datasets (see docs/model-cards/skin-v1.md):
  HAM10000 (Tschandl 2018, Harvard Dataverse doi:10.7910/DVN/DBW86T, CC BY-NC 4.0) - dermoscopy
  PAD-UFES-20 (Pacheco 2020, Mendeley Data doi:10.17632/zr7vgbcyr2.1, CC BY 4.0) - smartphone photos
Label mapping (3 tiers):
  urgent-referral: HAM mel, bcc | PAD MEL, BCC, SCC
  needs-review   : HAM akiec (actinic keratosis / Bowen) | PAD ACK
  benign         : HAM nv, bkl, df, vasc | PAD NEV, SEK
Group id for splitting: HAM lesion_id, PAD patient_id (prefixed by dataset so ids never collide).

Usage: python skin_prep.py /home/user/ml/data /home/user/ml/cache
"""
import io, sys, zipfile
from pathlib import Path
import numpy as np, pandas as pd
from PIL import Image
from skin_common import load_rgb, squash_u8

HAM_MAP = {"mel": 2, "bcc": 2, "akiec": 1, "nv": 0, "bkl": 0, "df": 0, "vasc": 0}
PAD_MAP = {"MEL": 2, "BCC": 2, "SCC": 2, "ACK": 1, "NEV": 0, "SEK": 0}


def main(data: Path, cache: Path):
    cache.mkdir(parents=True, exist_ok=True)
    rows, arrs = [], []
    ham = pd.read_csv(data / "ham/meta.tab", sep="\t")
    ham_lookup = {r.image_id: r for r in ham.itertuples()}
    for z in ("p1.zip", "p2.zip"):
        with zipfile.ZipFile(data / "ham" / z) as zf:
            for n in zf.namelist():
                if not n.lower().endswith(".jpg"):
                    continue
                iid = Path(n).stem
                r = ham_lookup[iid]
                arrs.append(squash_u8(load_rgb(io.BytesIO(zf.read(n)))))
                rows.append(dict(source="ham10000", image_id=iid, group="ham:" + r.lesion_id, dx=r.dx, y=HAM_MAP[r.dx]))
        print(z, len(rows), flush=True)
    pad = pd.read_csv(data / "pad/metadata.csv")
    pad_lookup = {r.img_id: r for r in pad.itertuples()}
    for i in (1, 2, 3):
        with zipfile.ZipFile(data / f"pad/imgs_part_{i}.zip") as zf:
            for n in zf.namelist():
                if not n.lower().endswith(".png"):
                    continue
                r = pad_lookup[Path(n).name]
                arrs.append(squash_u8(load_rgb(io.BytesIO(zf.read(n)))))
                rows.append(dict(source="pad-ufes-20", image_id=r.img_id, group="pad:" + r.patient_id, dx=r.diagnostic, y=PAD_MAP[r.diagnostic]))
        print("pad", i, len(rows), flush=True)
    pd.DataFrame(rows).to_csv(cache / "index.csv", index=False)
    np.save(cache / "images224.npy", np.stack(arrs))
    print("done", len(rows))


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
