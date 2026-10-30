"""Export backbone+head to ONE ONNX graph (input NCHW float32 ImageNet-normalised, output raw logits), quantize to int8
(weight-only per-channel int8), then report parity PyTorch vs ONNX fp32 vs int8 on TEST images.
Usage: python skin_export.py CACHE_DIR RUN_DIR OUT_DIR   (RUN_DIR has head.npz [+ backbone.pt if fine-tuned] and split.csv)"""
import json, sys, time
from pathlib import Path
import numpy as np, onnx, onnxruntime as ort, pandas as pd, torch, torch.nn as nn
from onnxruntime.quantization import CalibrationDataReader, QuantFormat, QuantType, quantize_static
from skin_common import S, to_tensor
from skin_features import backbone

torch.set_num_threads(4)
MIN_QUANT_ELEMS = 20000


class SkinNet(nn.Module):
    def __init__(self, bb, W, b):
        super().__init__()
        self.features, self.avgpool = bb.features, bb.avgpool
        self.head = nn.Linear(960, W.shape[0])
        self.head.weight.data = torch.from_numpy(W); self.head.bias.data = torch.from_numpy(b)

    def forward(self, x):
        return self.head(torch.flatten(self.avgpool(self.features(x)), 1))


def load_net(run: Path) -> SkinNet:
    h = np.load(run / "head.npz")
    bb = backbone()
    if (run / "backbone.pt").exists():
        bb.load_state_dict(torch.load(run / "backbone.pt"))
    return SkinNet(bb, h["W"], h["b"]).eval()


def weight_only_int8(src: Path, dst: Path):
    """Per-output-channel symmetric int8 weights + DequantizeLinear in the graph; activations stay fp32.
    (Static QDQ int8 of activations was tried first and rejected: MobileNetV3 lost too much, see docs/agents/training.md.)"""
    from onnx import TensorProto, helper, numpy_helper
    m = onnx.load(str(src))
    inits = {i.name: i for i in m.graph.initializer}
    new_nodes, drop = [], set()
    for n in m.graph.node:
        if n.op_type in ("Conv", "Gemm") and n.input[1] in inits:
            w = numpy_helper.to_array(inits[n.input[1]]).astype(np.float32)
            if n.op_type == "Gemm":  # weight is [out, in] with transB=1 in torch export
                axis = 0
            else:
                axis = 0
            if w.size < MIN_QUANT_ELEMS:  # tiny early layers are the most sensitive and cost almost nothing in size: keep fp32
                new_nodes.append(n)
                continue
            flat = w.reshape(w.shape[0], -1)
            scale = np.maximum(np.abs(flat).max(1), 1e-8) / 127.0
            q = np.clip(np.rint(flat / scale[:, None]), -127, 127).astype(np.int8).reshape(w.shape)
            wn = n.input[1]
            m.graph.initializer.append(numpy_helper.from_array(q, wn + "_q"))
            m.graph.initializer.append(numpy_helper.from_array(scale.astype(np.float32), wn + "_scale"))
            new_nodes.append(helper.make_node("DequantizeLinear", [wn + "_q", wn + "_scale"], [wn + "_dq"], axis=axis, name=wn + "_dql"))
            n.input[1] = wn + "_dq"
            drop.add(wn)
        new_nodes.append(n)
    keep = [i for i in m.graph.initializer if i.name not in drop]
    del m.graph.initializer[:]
    m.graph.initializer.extend(keep)
    del m.graph.node[:]
    m.graph.node.extend(new_nodes)
    onnx.checker.check_model(m)
    onnx.save(m, str(dst))


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
    imgs = np.load(cache / "images_all.npy", mmap_mode="r")
    sp = pd.read_csv(run / "split.csv")
    rng = np.random.default_rng(0)
    cal_idx = np.sort(rng.choice(np.where(sp.split == "train")[0], 300, replace=False))
    weight_only_int8(fp32, q)
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
