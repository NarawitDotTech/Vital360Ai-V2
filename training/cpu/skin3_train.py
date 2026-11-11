"""skin-v1.2: continue fine-tuning the skin-v1.1 weights (same blocks 7..16 + 20-class head, same data/augmentation/sampling as skin2_train.py)
with CutMix/MixUp (p 0.5), weight EMA, lower LR and more epochs. Epoch selection on the validation split (EMA weights).
Usage: python skin3_train.py CACHE2 V11_RUN_DIR RUN_DIR [epochs]"""
import copy, json, sys, threading, queue, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as Fn
from skin_calib import fit_temperature
from skin2_train import (NAMES, K, NC, SEED, SRC_W, HAM_NEVUS_W, EPOCH_DRAWS, augment, to_tensor_batch, Top, bn_eval, stage1_of, logits_of, macro_f1)
from skin_features import backbone
from skin_decide import TIER_IDX

torch.set_num_threads(4)
EMA_DECAY, MIX_P = 0.995, 0.5


def main(c2: Path, old_run: Path, run: Path, epochs=10):
    torch.manual_seed(SEED); rng = np.random.default_rng(SEED + 1)
    run.mkdir(parents=True, exist_ok=True)
    idx = pd.read_csv(c2 / "index2.csv"); idx.to_csv(run / "split.csv", index=False)
    y = idx["cls"].map({n: i for i, n in enumerate(NAMES)}).to_numpy()
    E = np.load(c2 / "eval224.npy", mmap_mode="r"); T = np.load(c2 / "train256.npy", mmap_mode="r")
    tr, va = np.where(idx.split == "train")[0], np.where(idx.split == "val")[0]
    hs = (idx.cls == "healthy-skin").to_numpy()
    w_s = np.array([HAM_NEVUS_W if (s == "ham10000" and c == "nevus") else SRC_W[(s, h)] for s, h, c in zip(idx.source, hs, idx.cls)])
    p_s = w_s[tr] / w_s[tr].sum()
    eff = np.bincount(y[tr], weights=p_s, minlength=NC) * EPOCH_DRAWS
    cw = torch.tensor((eff.sum() / (NC * np.maximum(eff, 1))) ** 0.5, dtype=torch.float32)
    bb = backbone(); bb.load_state_dict(torch.load(old_run / "backbone.pt"))
    front = nn.Sequential(*list(bb.features)[:K]).eval()
    model = Top(bb)
    h = np.load(old_run / "head.npz")
    with torch.no_grad():
        model.head.weight.copy_(torch.from_numpy(h["W"])); model.head.bias.copy_(torch.from_numpy(h["b"]))
    ema = copy.deepcopy(model)
    for p in ema.parameters(): p.requires_grad_(False)
    t0 = time.time()
    S1va = stage1_of(front, E[va])
    opt = torch.optim.AdamW([{"params": model.blocks.parameters(), "lr": 8e-5}, {"params": model.head.parameters(), "lr": 6e-4}], weight_decay=1e-2)
    steps_per = EPOCH_DRAWS // 64
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[8e-5, 6e-4], total_steps=epochs * steps_per, pct_start=0.1)
    pool = ThreadPoolExecutor(3)
    def make_batch(b, seed):
        r = np.random.default_rng(seed)
        return np.stack([augment(np.asarray(T[i]), r) for i in b])
    best = (-1, None, None)
    for ep in range(epochs):
        model.train(); bn_eval(model)
        draws = rng.choice(tr, size=steps_per * 64, p=p_s)
        q: "queue.Queue" = queue.Queue(maxsize=6)
        def producer():
            futs = []
            for s in range(steps_per):
                b = np.sort(draws[s * 64:(s + 1) * 64])
                futs.append((b, pool.submit(make_batch, b, int(rng.integers(0, 2 ** 31)))))
                while len(futs) > 5: q.put(futs.pop(0))
            for f in futs: q.put(f)
        threading.Thread(target=producer, daemon=True).start()
        tl = 0.0
        for s in range(steps_per):
            b, fut = q.get()
            x = to_tensor_batch(fut.result()); yb = torch.from_numpy(y[b]); yb2, lam = yb, 1.0
            if rng.random() < MIX_P:
                perm = torch.randperm(len(b)); yb2 = yb[perm]; lam = float(rng.beta(1.0, 1.0))
                if rng.random() < 0.5:  # CutMix
                    r = np.sqrt(1 - lam); ch, cw_ = int(224 * r), int(224 * r)
                    cy, cx = int(rng.integers(0, 224)), int(rng.integers(0, 224))
                    y0, y1, x0, x1 = max(cy - ch // 2, 0), min(cy + ch // 2, 224), max(cx - cw_ // 2, 0), min(cx + cw_ // 2, 224)
                    x = x.clone(); x[:, :, y0:y1, x0:x1] = x[perm][:, :, y0:y1, x0:x1]
                    lam = 1 - (y1 - y0) * (x1 - x0) / (224 * 224)
                else:  # MixUp (mild)
                    lam = max(lam, 1 - lam); x = lam * x + (1 - lam) * x[perm]
            with torch.no_grad(): s1 = front(x)
            out = model(s1)
            loss = lam * Fn.cross_entropy(out, yb, weight=cw, label_smoothing=0.05) + (1 - lam) * Fn.cross_entropy(out, yb2, weight=cw, label_smoothing=0.05)
            opt.zero_grad(); loss.backward(); opt.step(); sched.step(); tl += loss.item()
            with torch.no_grad():
                d = min(EMA_DECAY, (1 + ep * steps_per + s) / (10 + ep * steps_per + s))
                for pe, pm in zip(ema.parameters(), model.parameters()): pe.mul_(d).add_(pm.detach(), alpha=1 - d)
            if s % 50 == 0: print(f"  ep{ep} step {s}/{steps_per} loss {loss.item():.3f} ({time.time()-t0:.0f}s)", flush=True)
        pv = logits_of(ema, S1va).argmax(1); f1 = macro_f1(y[va], pv)
        tacc = float((TIER_IDX[pv] == TIER_IDX[y[va]]).mean())
        print(f"epoch {ep} loss {tl/steps_per:.3f} EMA val acc {(pv==y[va]).mean():.4f} macroF1 {f1:.4f} tier-acc {tacc:.4f} healthy-recall {(pv[hs[va]]==NAMES.index('healthy-skin')).mean():.3f} ({time.time()-t0:.0f}s)", flush=True)
        torch.save(ema.state_dict(), run / f"ema_ep{ep}.pt")
        if f1 > best[0]: best = (f1, copy.deepcopy(ema.state_dict()), ep)
    ema.load_state_dict(best[1]); print("best epoch", best[2], flush=True)
    Tm = fit_temperature(logits_of(ema, S1va), y[va])
    full = backbone()
    for i, l in enumerate(list(full.features)[K:]): l.load_state_dict(ema.blocks[i].state_dict())
    torch.save(full.state_dict(), run / "backbone.pt")
    np.savez(run / "head.npz", W=ema.head.weight.numpy(), b=ema.head.bias.numpy(), T=Tm)
    json.dump(dict(method="continue v1.1 fine-tune: cutmix/mixup p=0.5, EMA 0.995, LR 8e-5/6e-4", epochs=epochs, best_epoch=best[2], T=Tm, seed=SEED, train_seconds=time.time() - t0), open(run / "train_report.json", "w"), indent=1)
    print("T", Tm, flush=True)


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]), int(sys.argv[4]) if len(sys.argv) > 4 else 10)
