"""ONNX export, int8 (QDQ static) / fp16 conversion, parity check, size + latency report, manifest patch."""
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np
import torch


def export_onnx(model, example: torch.Tensor, path, opset: int = 17) -> None:
    """Contract with src/inference/ort.ts: ONE input (float32), ONE output = raw logits [N, K]. No softmax,
    no temperature (the app applies manifest temperature itself)."""
    model = model.eval().cpu().float()
    kw = dict(input_names=["input"], output_names=["logits"], opset_version=opset,
              dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}})
    try:
        torch.onnx.export(model, example.cpu().float(), str(path), dynamo=False, **kw)
    except TypeError:  # older torch without the dynamo kwarg
        torch.onnx.export(model, example.cpu().float(), str(path), **kw)


def _session(path):
    import onnxruntime as ort
    return ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])


def run_onnx(path, x: np.ndarray, bs: int = 32) -> np.ndarray:
    sess = _session(path)
    name = sess.get_inputs()[0].name
    return np.concatenate([sess.run(None, {name: x[i:i + bs].astype(np.float32)})[0] for i in range(0, len(x), bs)])


def quantize_int8(fp32_path, int8_path, calib: np.ndarray) -> None:
    """Static QDQ int8 quantisation using ~200 calibration inputs (no augmentation)."""
    from onnxruntime.quantization import CalibrationDataReader, QuantFormat, QuantType, quantize_static
    try:
        from onnxruntime.quantization.shape_inference import quant_pre_process
        pre = str(fp32_path).replace(".onnx", ".pre.onnx")
        quant_pre_process(str(fp32_path), pre, skip_symbolic_shape=True)
        src = pre
    except Exception as e:
        print("quant_pre_process skipped:", e)
        src = str(fp32_path)

    class Reader(CalibrationDataReader):
        def __init__(self):
            self.it = iter([{"input": c[None].astype(np.float32)} for c in calib])

        def get_next(self):
            return next(self.it, None)

    quantize_static(src, str(int8_path), Reader(), quant_format=QuantFormat.QDQ, per_channel=True,
                    weight_type=QuantType.QInt8, activation_type=QuantType.QUInt8)


def to_fp16(fp32_path, fp16_path) -> bool:
    try:
        import onnx
        from onnxconverter_common import float16
        m = onnx.load(str(fp32_path))
        onnx.save(float16.convert_float_to_float16(m, keep_io_types=True), str(fp16_path))
        return True
    except Exception as e:
        print("fp16 conversion failed:", e)
        return False


def sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def latency_ms(path, x1: np.ndarray, n: int = 50) -> float:
    """Median single-sample latency with ORT CPU (Kaggle CPU, NOT a phone: re-measure on a real phone)."""
    sess = _session(path)
    name = sess.get_inputs()[0].name
    for _ in range(5):
        sess.run(None, {name: x1})
    ts = []
    for _ in range(n):
        t = time.perf_counter()
        sess.run(None, {name: x1})
        ts.append((time.perf_counter() - t) * 1000)
    return float(np.median(ts))


def parity_report(torch_logits: np.ndarray, onnx_logits: np.ndarray):
    d = np.abs(torch_logits - onnx_logits)
    return {"max_abs_diff": float(d.max()), "mean_abs_diff": float(d.mean()),
            "top1_agreement": float((torch_logits.argmax(1) == onnx_logits.argmax(1)).mean())}


def manifest_patch(model_id, version, file_path, labels, temperature, threshold, extra_onnx=None):
    """The fields to paste into public/models/manifest.json for this model (tiers stay as already in the manifest)."""
    p = Path(file_path)
    patch = {"id": model_id, "version": version, "url": f"/models/{p.name}", "sha256": sha256(p),
             "bytes": os.path.getsize(p), "labels": labels, "threshold": round(threshold, 3),
             "temperature": round(temperature, 4)}
    if extra_onnx:
        patch["onnx"] = extra_onnx
    return patch


def dump(obj, path) -> None:
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
