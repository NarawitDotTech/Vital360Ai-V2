"""Export backbone+head to ONE ONNX graph (input NCHW float32 ImageNet-normalised, output raw logits), quantize to int8
(static QDQ, per-channel weights, calibrated on train images), then report parity PyTorch vs ONNX fp32 vs int8 on TEST images.
Usage: python skin_export.py CACHE_DIR RUN_DIR OUT_DIR   (RUN_DIR has head.npz [+ backbone.pt if fine-tuned] and split.csv)"""
import json, sys, time
from pathlib import Path
import numpy as np, onnx, onnxruntime as ort, pandas as pd, torch, torch.nn as nn
from onnxruntime.quantization import CalibrationDataReader, QuantFormat, QuantType, quantize_static
from skin_common import S, to_tensor
from skin_features import backbone

torch.set_num_threads(4)


class SkinNet(nn.Module):
    def __init__(self, bb, W, b):
        super().__init__()
        self.features, self.avgpool = bb.features, bb.avgpool
        self.head = nn.Linear(960, 3)
        self.head.weight.data = torch.from_numpy(W); self.head.bias.data = torch.from_numpy(b)

    def forward(self, x):
        return self.head(torch.flatten(self.avgpool(self.features(x)), 1))


def load_net(run: Path) -> SkinNet:
    h = np.load(run / "head.npz")
    bb = backbone()
    if (run / "backbone.pt").exists():
        bb.load_state_dict(torch.load(run / "backbone.pt"))
    return SkinNet(bb, h["W"], h["b"]).eval()


class Reader(CalibrationDataReader):
    def __init__(self, arrs):
        self.it = iter([{"input": a[None]} for a in arrs])

    def get_next(self):
        return next(self.it, None)


def main(cache: Path, run: Path, out: Path):
    out.mkdir(parents=True, exist_ok=True)
    net = load_net(run)
    fp32, pre, q = out / "skin-v1.fp32.onnx", out / "skin-v1.pre.onnx", out / "skin-v1.int8.onnx"
    torch.onnx.export(net, torch.zeros(1, 3, S, S), str(fp32), input_names=["input"], output_names=["logits"], opset_version=17, dynamo=False)
    imgs = np.load(cache / "images224.npy", mmap_mode="r")
    sp = pd.read_csv(run / "split.csv")
    rng = np.random.default_rng(0)
    cal_idx = np.sort(rng.choice(np.where(sp.split == "train")[0], 300, replace=False))
    from onnxruntime.quantization.shape_inference import quant_pre_process
    quant_pre_process(str(fp32), str(pre))
    quantize_static(str(pre), str(q), Reader([to_tensor(imgs[i]) for i in cal_idx]), quant_format=QuantFormat.QDQ,
                    activation_type=QuantType.QUInt8, weight_type=QuantType.QInt8, per_channel=True)
    te = np.where(sp.split == "test")[0]
    xs = np.stack([to_tensor(imgs[i]) for i in te])
    with torch.no_grad():
        ref = np.concatenate([net(torch.from_numpy(xs[i:i + 64])).numpy() for i in range(0, len(xs), 64)])
    res = {}
    for name, path in (("fp32", fp32), ("int8", q)):
        s = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        t0 = time.time()
        lg = np.concatenate([s.run(None, {"input": xs[i:i + 1]})[0] for i in range(len(xs))])
        res[name] = lg
        res[name + "_ms"] = (time.time() - t0) / len(xs) * 1000
    np.save(out / "test_logits_torch.npy", ref); np.save(out / "test_logits_int8.npy", res["int8"])
    rep = dict(n_test=len(te), size_fp32=fp32.stat().st_size, size_int8=q.stat().st_size,
               fp32_vs_torch_max_abs=float(abs(res["fp32"] - ref).max()), fp32_vs_torch_top1=float((res["fp32"].argmax(1) == ref.argmax(1)).mean()),
               int8_vs_fp32_max_abs=float(abs(res["int8"] - res["fp32"]).max()), int8_vs_fp32_mean_abs=float(abs(res["int8"] - res["fp32"]).mean()),
               int8_vs_fp32_top1=float((res["int8"].argmax(1) == res["fp32"].argmax(1)).mean()),
               int8_ms_per_img_desktop_cpu=res["int8_ms"], fp32_ms_per_img_desktop_cpu=res["fp32_ms"])
    json.dump(rep, open(out / "parity.json", "w"), indent=1)
    print(json.dumps(rep, indent=1))
    onnx.checker.check_model(onnx.load(str(q)))


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]))
