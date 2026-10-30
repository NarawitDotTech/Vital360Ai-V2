"""Small CNN on the 501x64 log-mel canvas (SPRSound record-level normal / wheeze / crackle). Patient-level split, class-weighted CE,
temperature scaling + uncertainty threshold on val, ONNX export with in-graph normalisation, parity and TEST metrics.
Usage: python audio_train.py CACHE_AUDIO_DIR RUN_DIR OUT_METRICS_JSON REPO_DIR"""
import copy, datetime, hashlib, json, sys, time
from pathlib import Path
import numpy as np, onnxruntime as ort, pandas as pd, torch, torch.nn as nn, torch.nn.functional as Fn
from scipy.special import softmax
from sklearn.model_selection import StratifiedGroupKFold
from skin_calib import fit_temperature
from skin_metrics import bootstrap_generic, confusion, summarize

torch.set_num_threads(4)
SEED = 42
W_POW = float(__import__("os").environ.get("W_POW", "0.5"))
CLASSES = ["normal", "wheeze", "crackle"]
PAD = float(np.log(1e-10))


class Net(nn.Module):
    def __init__(self, mean, std):
        super().__init__()
        self.register_buffer("mean", torch.tensor(mean)); self.register_buffer("std", torch.tensor(std))
        ch = [1, 16, 32, 64, 96]
        self.convs = nn.Sequential(*[nn.Sequential(nn.Conv2d(ch[i], ch[i + 1], 3, 2, 1, bias=False), nn.BatchNorm2d(ch[i + 1]), nn.ReLU()) for i in range(4)])
        self.drop = nn.Dropout(0.3); self.fc = nn.Linear(192, 3)

    def forward(self, x):  # x [B,1,501,64] raw log-mel
        h = self.convs((x - self.mean) / self.std)
        h = torch.cat([h.mean((2, 3)), h.amax((2, 3))], 1)
        return self.fc(self.drop(h))


def augment(x, valid, rng):
    x = x.clone()
    for i in range(len(x)):
        v = int(valid[i])
        x[i, :, :v] += float(rng.uniform(-2.5, 2.5))  # gain (dB-ish in log-power)
        for _ in range(2):
            f = int(rng.integers(0, 10)); f0 = int(rng.integers(0, 64 - f)); x[i, :, :, f0:f0 + f] = x[i, :, :v].mean()
            t = int(rng.integers(0, 40)); t0 = int(rng.integers(0, max(1, v - t))); x[i, :, t0:t0 + t] = x[i, :, :v].mean()
    return x


def main(cache, run, out_json, repo):
    run.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(SEED); rng = np.random.default_rng(SEED)
    X = np.load(cache / "audio_mel.npy"); idx = pd.read_csv(cache / "audio_index.csv"); y = idx.y.to_numpy()
    valid = (X[:, :, 0] > PAD + 1e-3).sum(1)
    split = np.array(["train"] * len(idx), dtype=object)
    folds = list(StratifiedGroupKFold(n_splits=7, shuffle=True, random_state=SEED).split(idx, y, idx.group))
    split[folds[0][1]] = "test"; split[folds[1][1]] = "val"
    idx["split"] = split; idx.to_csv(run / "audio_split.csv", index=False)
    for a, b in (("train", "test"), ("val", "test"), ("train", "val")):
        assert not (set(idx[idx.split == a].group) & set(idx[idx.split == b].group))
    tr, va, te = (np.where(split == s)[0] for s in ("train", "val", "test"))
    real = np.concatenate([X[i, :valid[i]].ravel() for i in tr])
    net = Net(float(real.mean()), float(real.std()))
    cnt = np.bincount(y[tr], minlength=3); w = torch.tensor((cnt.sum() / (3 * cnt)) ** W_POW, dtype=torch.float32)
    EP = 30
    opt = torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=1e-2)
    steps = EP * (len(tr) // 32)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=3e-3, total_steps=steps, pct_start=0.2)
    Xt = torch.from_numpy(X)[:, None]

    def logits(ids):
        net.eval()
        with torch.no_grad():
            return np.concatenate([net(Xt[ids[i:i + 64]]).numpy() for i in range(0, len(ids), 64)])
    best = (-1, None, None); t0 = time.time()
    for ep in range(EP):
        net.train(); perm = rng.permutation(tr)
        for i in range(0, len(perm) - 31, 32):
            b = perm[i:i + 32]
            loss = Fn.cross_entropy(net(augment(Xt[b], valid[b], rng)), torch.from_numpy(y[b]), weight=w, label_smoothing=0.05)
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        s = summarize(confusion(y[va], logits(va).argmax(1)))
        f1 = float(s["f1"].mean())
        if ep % 3 == 2 or ep == EP - 1:
            print(f"epoch {ep} loss {loss.item():.3f} val acc {s['accuracy']:.3f} macroF1 {f1:.3f} ({time.time()-t0:.0f}s)", flush=True)
        if f1 > best[0]:
            best = (f1, copy.deepcopy(net.state_dict()), ep)
    net.load_state_dict(best[1]); net.eval()
    lv = logits(va); T = fit_temperature(lv, y[va])
    pv = softmax(lv / T, axis=1); base = float((pv.argmax(1) == y[va]).mean()); thr = 0.5
    for t in np.arange(0.34, 0.95, 0.01):
        m = pv.max(1) >= t
        if m.mean() < 0.6: break
        if (pv.argmax(1)[m] == y[va][m]).mean() >= base + 0.05: thr = float(round(t, 2)); break
    # export (fp32; the model is ~0.4 MB so int8 would only add risk)
    onnx_path = run / "audio-v1.onnx"
    torch.onnx.export(net, torch.zeros(1, 1, 501, 64), str(onnx_path), input_names=["input"], output_names=["logits"], opset_version=17, dynamo=False)
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    ref = logits(te); t1 = time.time()
    got = np.concatenate([sess.run(None, {"input": X[i][None, None]})[0] for i in te]); ms = (time.time() - t1) / len(te) * 1000
    parity = dict(max_abs_diff=float(abs(got - ref).max()), top1_agreement=float((got.argmax(1) == ref.argmax(1)).mean()), n=len(te))
    # test metrics (uncertain handled separately: coverage + accuracy on answered)
    p = softmax(got / T, axis=1); pred = p.argmax(1); yt = y[te]; cm = confusion(yt, pred, 3); s = summarize(cm)
    ans = p.max(1) >= thr
    present = np.bincount(yt, minlength=3) > 0
    ci = bootstrap_generic(idx.group.to_numpy()[te], lambda ix: dict(accuracy=float((pred[ix] == yt[ix]).mean()), macro_f1=float(summarize(confusion(yt[ix], pred[ix], 3))["f1"].mean())))
    now = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")
    per = {c: dict(sensitivity=float(s["sens"][i]), specificity=float(s["spec"][i])) for i, c in enumerate(CLASSES)}
    dash = dict(schemaVersion=1, isSample=False, generatedAt=now, source="measured", dataset="SPRSound (CC BY 4.0), paediatric stethoscope recordings", date=now[:10],
                description="MEASURED on a held-out patient-level test split of SPRSound (record-level labels). audio-v1 fp32 ONNX baseline for a school project; paediatric stethoscope audio, NOT validated on phone-microphone recordings.",
                splits={k: int((split == k).sum()) for k in ("train", "val", "test")},
                models=dict(audio=dict(label="Breath sound (audio-v1, measured)", classes=CLASSES, accuracy=float(s["accuracy"]), sensitivity=float(s["sens"][present].mean()), specificity=float(s["spec"][present].mean()),
                                       f1=float(s["f1"][present].mean()), perClass=per, confusionMatrix=cm.tolist(), latencyMs=round(ms, 1), sizeBytes=onnx_path.stat().st_size,
                                       testSetDescription=f"Held-out TEST: {len(te)} recordings from {idx.group.iloc[te].nunique()} patients (SPRSound, 10 s canvases). Latency = desktop CPU onnxruntime, not a phone.")),
                uncertain=dict(threshold=thr, coverage=float(ans.mean()), accuracy_answered=float((pred[ans] == yt[ans]).mean())), ci95=ci, parity=parity, temperature=T, bestEpoch=best[2], valMacroF1=best[0])
    json.dump(dash, open(out_json, "w"), indent=1)
    np.savez(run / "audio_test.npz", logits=got, te=te)
    sha = hashlib.sha256(onnx_path.read_bytes()).hexdigest()
    json.dump(dict(T=T, thr=thr, sha256=sha, bytes=onnx_path.stat().st_size, mean=float(net.mean), std=float(net.std)), open(run / "audio_report.json", "w"))
    print(json.dumps(dash, indent=1)[:3500])


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]), Path(sys.argv[4]))
