"""Shared skin preprocessing. MUST match src/inference/ort.ts (skinTensor) exactly.

Pipeline (same as the app): decode -> (browser) downscale so max side <= 640 -> RGB uint8 ->
area-weighted squash resize to SxS (no crop) -> /255 -> ImageNet mean/std -> NCHW float32.
"""
import numpy as np
from PIL import Image

S = 224
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
MAX_SIDE = 640  # src/capture/camera.ts maxSide
LABELS = ["benign", "needs-review", "urgent-referral"]


def area_matrix(n: int, s: int) -> np.ndarray:
    """(s, n) matrix: row i holds the fractional overlap of each source pixel with output cell i, / cell width."""
    w = np.zeros((s, n), dtype=np.float64)
    scale = n / s
    for i in range(s):
        a, b = i * scale, (i + 1) * scale
        for j in range(int(np.floor(a)), min(n, int(np.ceil(b)))):
            w[i, j] = (min(b, j + 1) - max(a, j)) / scale
    return w


def downscale_like_app(img: Image.Image, max_side: int = MAX_SIDE) -> Image.Image:
    """Stand-in for the canvas downscale in camera.ts (the browser uses bilinear; we use PIL BILINEAR with reducing_gap)."""
    w, h = img.size
    k = min(1.0, max_side / max(w, h))
    if k < 1.0:
        img = img.resize((max(1, round(w * k)), max(1, round(h * k))), Image.BILINEAR, reducing_gap=2.0)
    return img


def squash_u8(rgb: np.ndarray, s: int = S) -> np.ndarray:
    """uint8 HxWx3 -> uint8 sxsx3 via area-weighted resize (rounded). Used for the training cache."""
    return np.clip(np.rint(squash_f(rgb, s)), 0, 255).astype(np.uint8)


def squash_f(rgb: np.ndarray, s: int = S) -> np.ndarray:
    h, w, _ = rgb.shape
    wy, wx = area_matrix(h, s), area_matrix(w, s)
    x = rgb.astype(np.float64)
    out = np.einsum("ih,hwc->iwc", wy, x)
    return np.einsum("jw,iwc->ijc", wx, out)


def to_tensor(rgb_squashed: np.ndarray) -> np.ndarray:
    """float (not rounded) or uint8 sxsx3 in 0..255 -> float32 [3,s,s] normalised."""
    x = (rgb_squashed.astype(np.float32) / 255.0 - MEAN) / STD
    return np.ascontiguousarray(x.transpose(2, 0, 1), dtype=np.float32)


def load_rgb(path_or_file) -> np.ndarray:
    img = Image.open(path_or_file).convert("RGB")
    return np.asarray(downscale_like_app(img))
