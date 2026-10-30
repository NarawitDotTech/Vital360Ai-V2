"""Extract frozen MobileNetV3-Large (ImageNet) features. Usage: python skin_features.py CACHE_DIR
Train images get 3 views (orig, hflip, vflip) for cheap augmentation; val/test only the original view."""
import sys, time
from pathlib import Path
import numpy as np, torch, torchvision
from skin_common import MEAN, STD

torch.set_num_threads(4)


def backbone():
    m = torchvision.models.mobilenet_v3_large(weights=torchvision.models.MobileNet_V3_Large_Weights.IMAGENET1K_V1)
    return m.eval()


def prep(batch_u8: np.ndarray) -> torch.Tensor:
    x = (torch.from_numpy(batch_u8).float() / 255.0 - torch.from_numpy(MEAN)) / torch.from_numpy(STD)
    return x.permute(0, 3, 1, 2).contiguous()


@torch.no_grad()
def feats(m, x):
    f = torch.flatten(m.avgpool(m.features(x)), 1)  # 960
    return f.numpy()


def main(cache: Path):
    imgs = np.load(cache / "images224.npy", mmap_mode="r")
    m = backbone()
    views = {"orig": lambda x: x, "hflip": lambda x: x.flip(3), "vflip": lambda x: x.flip(2)}
    for name, fn in views.items():
        out = []
        t0 = time.time()
        for i in range(0, len(imgs), 64):
            out.append(feats(m, fn(prep(np.asarray(imgs[i:i + 64])))))
            if (i // 64) % 20 == 0:
                print(name, i, f"{time.time() - t0:.0f}s", flush=True)
        np.save(cache / f"feat_{name}.npy", np.concatenate(out))


if __name__ == "__main__":
    main(Path(sys.argv[1]))
