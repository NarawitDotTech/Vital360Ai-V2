"""Export skin-v1.1 (backbone+20-class head) to one ONNX graph, int8 per-channel weights (same recipe as skin_export.py), and write
fp32-torch / fp32-ONNX / int8-ONNX logits for the val and test images (224 px eval cache = exact app pipeline).
Usage: python skin2_export.py CACHE2 RUN_DIR OUT_DIR"""
import json, sys, time
from pathlib import Path
import numpy as np, onnx, onnxruntime as ort, pandas as pd, torch
from skin_common import S, to_tensor
from skin_export import load_net, weight_only_int8

torch.set_num_threads(4)


def main(c2: Path, run: Path, out: Path):
    out.mkdir(parents=True, exist_ok=True)
    net = load_net(run)
    fp32, q = out / "skin-v1.1.fp32.onnx", out / "skin-v1.1.int8.onnx"
    torch.onnx.export(net, torch.zeros(1, 3, S, S), str(fp32), input_names=["input"], output_names=["logits"], opset_version=17, dynamo=False)
    weight_only_int8(fp32, q)
    E = np.load(c2 / "eval224.npy", mmap_mode="r")
    idx = pd.read_csv(run / "split.csv")
    sess = {n: ort.InferenceSession(str(p), providers=["CPUExecutionProvider"]) for n, p in (("fp32", fp32), ("int8", q))}
    rep = dict(size_fp32=fp32.stat().st_size, size_int8=q.stat().st_size)
    for split in ("val", "test"):
        ii = np.where(idx.split == split)[0]
        xs = np.stack([to_tensor(E[i]) for i in ii])
        with torch.no_grad():
            ref = np.concatenate([net(torch.from_numpy(xs[i:i + 64])).numpy() for i in range(0, len(xs), 64)])
        res = {}
        for n, s in sess.items():
            t0 = time.time()
            res[n] = np.concatenate([s.run(None, {"input": xs[i:i + 1]})[0] for i in range(len(xs))])
            res[n + "_ms"] = (time.time() - t0) / len(xs) * 1000
        np.save(out / f"{split}_logits_torch.npy", ref); np.save(out / f"{split}_logits_int8.npy", res["int8"])
        rep[split] = dict(n=len(ii), fp32_vs_torch_max_abs=float(abs(res["fp32"] - ref).max()), int8_vs_fp32_max_abs=float(abs(res["int8"] - res["fp32"]).max()),
                          int8_vs_fp32_mean_abs=float(abs(res["int8"] - res["fp32"]).mean()), int8_vs_fp32_top1=float((res["int8"].argmax(1) == res["fp32"].argmax(1)).mean()))
        rep["int8_ms_per_img_desktop_cpu"] = res["int8_ms"]; rep["fp32_ms_per_img_desktop_cpu"] = res["fp32_ms"]
    assert rep["size_int8"] <= 15 * 1024 * 1024, rep["size_int8"]
    json.dump(rep, open(out / "parity.json", "w"), indent=1)
    print(json.dumps(rep, indent=1))
    onnx.checker.check_model(onnx.load(str(q)))


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]))
