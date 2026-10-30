"""Generic resumable training loop (AMP, checkpoint every epoch, TensorBoard, early stopping)."""
import time
from pathlib import Path

import numpy as np
import torch

from .ckpt import load_if_exists, save_atomic
from .metrics import confusion, summary

DEV = "cuda" if torch.cuda.is_available() else "cpu"


@torch.no_grad()
def predict(model, dl, prep):
    """Return (logits [N,K], labels [N]) in dataloader order (use shuffle=False)."""
    model.eval()
    zs, ys = [], []
    for x, y in dl:
        x = prep(x.to(DEV, non_blocking=True), train=False)
        with torch.autocast(device_type=DEV, dtype=torch.float16, enabled=DEV == "cuda"):
            z = model(x)
        zs.append(z.float().cpu().numpy())
        ys.append(y.numpy())
    return np.concatenate(zs), np.concatenate(ys)


def fit(model, train_dl, val_dl, crit, opt, sched, epochs, out_dir, prep, n_classes, writer=None, patience=6):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    model.to(DEV)
    scaler = torch.amp.GradScaler("cuda", enabled=DEV == "cuda")
    start, best, bad = 0, -1.0, 0
    ck = load_if_exists(out / "last.pt")
    if ck:  # RESUME after a disconnect / restart
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"])
        scaler.load_state_dict(ck["scaler"])
        start, best, bad = ck["epoch"] + 1, ck["best"], ck["bad"]
        print(f"Resumed from epoch {ck['epoch']} (best val macro-F1 {best:.4f})")
    for ep in range(start, epochs):
        t0 = time.time()
        model.train()
        tot, n = 0.0, 0
        for x, y in train_dl:
            x = prep(x.to(DEV, non_blocking=True), train=True)
            y = y.to(DEV, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=DEV, dtype=torch.float16, enabled=DEV == "cuda"):
                loss = crit(model(x), y)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(opt)
            scaler.update()
            tot += loss.item() * len(y)
            n += len(y)
        sched.step()
        z, yv = predict(model, val_dl, prep)
        s, _ = summary(confusion(yv, z.argmax(1), n_classes))
        f1 = s["f1"]
        if writer:
            writer.add_scalar("train/loss", tot / n, ep)
            writer.add_scalar("val/macro_f1", f1, ep)
            writer.add_scalar("val/acc", s["accuracy"], ep)
            writer.add_scalar("lr", opt.param_groups[0]["lr"], ep)
        improved = f1 > best
        if improved:
            best, bad = f1, 0
            save_atomic(model.state_dict(), out / "best.pt")
        else:
            bad += 1
        save_atomic({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                     "scaler": scaler.state_dict(), "epoch": ep, "best": best, "bad": bad}, out / "last.pt")
        print(f"epoch {ep + 1}/{epochs} loss {tot / n:.4f} val macroF1 {f1:.4f} acc {s['accuracy']:.4f} "
              f"{'*' if improved else ''} ({time.time() - t0:.0f}s)")
        if bad >= patience:
            print("early stop")
            break
    model.load_state_dict(torch.load(out / "best.pt", map_location=DEV))
    return best
