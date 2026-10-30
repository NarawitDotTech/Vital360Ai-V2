"""skin-v2 input pipeline (Python side). MUST match src/capture/camera.ts (cropSquare / guideRect) + src/inference/skin-preprocess.ts.

App pipeline: decode (EXIF applied by the browser) -> square crop (camera: the on-screen guide box = 0.70 x the visible frame height,
centred; file upload: centred square of the short side) -> if the square is larger than 448 px, shrink it to 448 px (browser: repeated
<=2x canvas draws, ~box filter; here PIL BOX) -> area-weighted resize to 224x224 -> /255 -> ImageNet mean/std -> NCHW float32.
Training uses the same crop, plus a 256 px copy of the square for scale/crop jitter.
"""
import numpy as np
from PIL import Image, ImageOps
from scipy.ndimage import uniform_filter
from skin_common import squash_u8

SQ = 448       # max side of the intermediate square (canvas in the browser)
TRAIN_S = 256  # training cache side


def open_rgb(file) -> Image.Image:
    """Decode like createImageBitmap(file): EXIF orientation applied. Alpha is dropped (all PAD-UFES-20 alphas are opaque)."""
    im = Image.open(file)
    im = ImageOps.exif_transpose(im)
    return im.convert("RGB")


def center_square(im: Image.Image, frac: float = 1.0) -> Image.Image:
    w, h = im.size
    s = max(1, int(round(min(w, h) * frac)))
    x, y = (w - s) // 2, (h - s) // 2
    return im.crop((x, y, x + s, y + s))


def shrink(sq: Image.Image, side: int) -> Image.Image:
    return sq.resize((side, side), Image.BOX) if sq.size[0] > side else sq


def eval224(sq: Image.Image) -> np.ndarray:
    """Exact (Python) version of what the app feeds the network, as uint8 224x224x3."""
    return squash_u8(np.asarray(shrink(sq, SQ)))


def train256(sq: Image.Image) -> np.ndarray:
    s = shrink(sq, TRAIN_S)
    if s.size[0] != TRAIN_S:
        s = s.resize((TRAIN_S, TRAIN_S), Image.BICUBIC)
    return np.asarray(s)


# ---- healthy-skin patch sampling and automatic checks ----
def patch_stats(sq: Image.Image) -> dict:
    a = np.asarray(sq.resize((96, 96), Image.BOX)).astype(np.float32)
    r, g, b = a[..., 0], a[..., 1], a[..., 2]
    Y = 0.299 * r + 0.587 * g + 0.114 * b
    Cb = 128 - 0.168736 * r - 0.331264 * g + 0.5 * b
    Cr = 128 + 0.5 * r - 0.418688 * g - 0.081312 * b
    skin = (Cr >= 136) & (Cr <= 178) & (Cb >= 77) & (Cb <= 124) & (Y > 30)
    m = float(np.median(Y))
    bY, bCr = uniform_filter(Y, 9), uniform_filter(Cr, 9)
    spots = float(((Cr - uniform_filter(Cr, 21)) > 8).mean())  # small red spots (rash, acne, petechiae)
    inner = (slice(5, -5), slice(5, -5))  # ignore the filter border
    return dict(skin=float(skin.mean()), medY=m, dark_blob=float(bY[inner].min() / max(m, 1)), bright_blob=float(bY[inner].max() / max(m, 1)),
                red_blob=float(bCr[inner].max() - np.median(Cr)), pale_blob=float(np.median(Cr) - bCr[inner].min()),
                dark_px=float((Y < 0.6 * m).mean()), spots=spots, hue=float((np.median(Cr) - 128) / max(128 - np.median(Cb), 1)), cv=float(Y.std() / max(m, 1)))


def patch_ok(st: dict) -> bool:
    return (st["skin"] >= 0.93 and 45 <= st["medY"] <= 235 and st["dark_blob"] >= 0.80 and st["bright_blob"] <= 1.30
            and st["red_blob"] <= 5.5 and st["pale_blob"] <= 7.0 and st["dark_px"] <= 0.008 and st["spots"] <= 0.0015 and 0.55 <= st["hue"] <= 2.0 and st["cv"] <= 0.12)


def healthy_patches(im: Image.Image, rng: np.random.Generator, n_max: int, tries: int = 14, frac=(0.28, 0.45), excl: float = 0.2):
    """Random square patches (side = frac * short side) that stay outside a central disc of radius excl*short side (the lesion area)
    and pass patch_ok. Returns [(Image, stats)]. Tiny sources (< 160 px short side) give nothing."""
    w, h = im.size
    m = min(w, h)
    out = []
    if m < 160:
        return out
    n_eval = 0
    for _ in range(300):
        if n_eval >= tries:
            break
        s = int(m * rng.uniform(*frac))
        x, y = int(rng.integers(0, w - s + 1)), int(rng.integers(0, h - s + 1))
        # nearest point of the patch to the image centre must be outside the exclusion disc
        nx, ny = min(max(w / 2, x), x + s), min(max(h / 2, y), y + s)
        if (nx - w / 2) ** 2 + (ny - h / 2) ** 2 < (excl * m) ** 2:
            continue
        p = im.crop((x, y, x + s, y + s))
        n_eval += 1
        st = patch_stats(p)
        if patch_ok(st):
            out.append((p, st))
            if len(out) >= n_max:
                break
    return out
