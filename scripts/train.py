"""Train a neural survival model and evaluate it on the held-out test days.

  oft  : Order-Flow Transformer (event tokens + queue tokens + context features)
  snap : snapshot conv-Transformer baseline (B4: last 60 s of top-10 L2 levels + L2 features)

Run:  .venv/bin/python scripts/train.py --model oft --pretrain-epochs 1 --epochs 6 --tag oft_pt
Writes results/<tag>.json (metrics), results/<tag>_curve.json (learning curves),
results/surv_<tag>.npy (test survival curves) and checkpoints/<tag>.pt.
"""

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from fillprob.bins import discrete_nll, surv_from_logits, to_bins  # noqa: E402
from fillprob.data import GLOBAL_FEATURES, L2_FEATURES, SPLITS, Standardizer, load_days  # noqa: E402
from fillprob.evaluate import evaluate  # noqa: E402
from fillprob.metrics import antolini_cindex  # noqa: E402
from fillprob.models import OrderFlowTransformer, SnapshotConvTransformer, n_params  # noqa: E402

RES, CKPT = ROOT / "results", ROOT / "checkpoints"


def batches(n, bs, shuffle, rng):
    idx = rng.permutation(n) if shuffle else np.arange(n)
    for i in range(0, n, bs):
        yield np.sort(idx[i:i + bs]) if shuffle else idx[i:i + bs]


def to_dev(a, idx, dev, dtype=torch.float32):
    return torch.from_numpy(np.ascontiguousarray(a[idx])).to(dev, dtype)


@torch.no_grad()
def predict(model, kind, split, g, dev, bs=1024):
    model.eval()
    out = []
    for idx in batches(len(g), bs, False, None):
        if kind == "oft":
            z = model(to_dev(split.events, idx, dev), to_dev(split.queue, idx, dev), to_dev(g, idx, dev))
        else:
            z = model(to_dev(split.snaps, idx, dev), to_dev(g, idx, dev))
        out.append(z.float().cpu())
    return torch.cat(out)


def cosine(step, total, warmup=200):
    if step < warmup:
        return step / warmup
    return 0.5 * (1 + math.cos(math.pi * min(1.0, (step - warmup) / max(1, total - warmup))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["oft", "snap"], default="oft")
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--pretrain-epochs", type=int, default=0)
    ap.add_argument("--stride", type=int, default=4)
    ap.add_argument("--test-stride", type=int, default=3)
    ap.add_argument("--bs", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--d", type=int, default=128)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--wd", type=float, default=0.01)
    ap.add_argument("--train-frac", type=float, default=1.0, help="fraction of training labels (learning curves)")
    ap.add_argument("--no-l3-globals", action="store_true")
    ap.add_argument("--no-events", action="store_true", help="ablation: drop all event tokens")
    ap.add_argument("--no-queue", action="store_true", help="ablation: drop all queue tokens")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    tag = a.tag or a.model
    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    dev = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    RES.mkdir(exist_ok=True)
    CKPT.mkdir(exist_ok=True)

    t0 = time.time()
    tr = load_days(SPLITS["train"], stride=a.stride)
    va = load_days(SPLITS["val"], stride=5)
    te = load_days(SPLITS["test"], stride=a.test_stride)
    if a.train_frac < 1.0:
        keep = np.sort(rng.choice(len(tr.df), int(len(tr.df) * a.train_frac), replace=False))
        tr.df, tr.events, tr.queue, tr.snaps = tr.df.iloc[keep].reset_index(drop=True), tr.events[keep], tr.queue[keep], tr.snaps[keep]
    print(f"loaded train {len(tr.df)} val {len(va.df)} test {len(te.df)} in {time.time() - t0:.0f}s", flush=True)
    for sp in (tr, va, te):              # ablations: blank a token stream (padding everywhere)
        if a.no_events:
            sp.events[..., 0] = 0
        if a.no_queue:
            sp.queue[..., 5] = 0

    cols = L2_FEATURES if (a.model == "snap" or a.no_l3_globals) else GLOBAL_FEATURES
    sc = Standardizer(cols).fit(tr.df)
    gtr, gva, gte = sc(tr.df), sc(va.df), sc(te.df)
    btr = to_bins(tr.df.duration.to_numpy(), tr.df.event.to_numpy())
    bva = to_bins(va.df.duration.to_numpy(), va.df.event.to_numpy())

    if a.model == "oft":
        model = OrderFlowTransformer(len(cols), d=a.d, layers=a.layers, dropout=a.dropout)
        sub = tr.events[rng.choice(len(tr.events), min(20000, len(tr.events)), replace=False)].astype(np.float32)
        real = sub[..., 0] > 0
        ev_num = sub[..., 1:][real]
        qs = tr.queue[rng.choice(len(tr.queue), min(20000, len(tr.queue)), replace=False)].astype(np.float32)
        q_num = qs[..., :5][qs[..., 5] > 0]
        model.set_token_stats(ev_num.mean(0), ev_num.std(0) + 1e-3, q_num.mean(0), q_num.std(0) + 1e-3)
    else:
        model = SnapshotConvTransformer(len(cols), d=a.d, layers=a.layers, dropout=a.dropout)
        s = tr.snaps[rng.choice(len(tr.snaps), min(20000, len(tr.snaps)), replace=False)].astype(np.float32).reshape(-1, 40)
        model.set_snap_stats(s.mean(0), s.std(0) + 1e-3)
    # start the hazard head at the empirical (training) discrete hazards, so training starts from the KM curve
    k_idx, k_ev = btr
    at_risk = np.array([(k_idx >= k).sum() for k in range(20)], dtype=np.float64)
    events_k = np.bincount(k_idx[k_ev == 1], minlength=20).astype(np.float64)
    h0 = np.clip(events_k / np.maximum(at_risk, 1), 1e-4, 1 - 1e-4)
    with torch.no_grad():
        model.head[-1].bias.copy_(torch.tensor(np.log(h0 / (1 - h0)), dtype=torch.float32))
        model.head[-1].weight.mul_(0.1)
    model.to(dev)
    print(f"{tag}: {n_params(model):,} trainable parameters (excluding pretraining heads)", flush=True)
    curve = {"tag": tag, "n_params": n_params(model), "pretrain": [], "train": [], "val": []}

    def run_epochs(n_epochs, lr, pretrain):
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=a.wd)
        steps_per_epoch = math.ceil(len(tr.df) / a.bs)
        total, step = n_epochs * steps_per_epoch, 0
        best = (float("inf"), None)
        for ep in range(n_epochs):
            model.train()
            run, n_run, t_ep = 0.0, 0, time.time()
            for idx in batches(len(tr.df), a.bs, True, rng):
                for gr in opt.param_groups:
                    gr["lr"] = lr * cosine(step, total)
                if a.model == "oft":
                    ev, q, g = to_dev(tr.events, idx, dev), to_dev(tr.queue, idx, dev), to_dev(gtr, idx, dev)
                    if pretrain:
                        loss, info = model.pretrain_loss(ev, q, g)
                    else:
                        loss = discrete_nll(model(ev, q, g), btr[0][idx], btr[1][idx])
                else:
                    loss = discrete_nll(model(to_dev(tr.snaps, idx, dev), to_dev(gtr, idx, dev)), btr[0][idx], btr[1][idx])
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                step += 1
                run += loss.item()
                n_run += 1
                if step % 200 == 0:
                    rec = {"step": step, "epoch": ep + n_run / steps_per_epoch, "loss": run / n_run}
                    if pretrain:
                        rec.update(info)
                    curve["pretrain" if pretrain else "train"].append(rec)
                    print(f"{'pre' if pretrain else 'sup'} ep {ep} step {step}/{total} loss {run / n_run:.4f} ({time.time() - t_ep:.0f}s)", flush=True)
                    run, n_run = 0.0, 0
            if pretrain:
                continue
            z = predict(model, a.model, va, gva, dev)
            vnll = discrete_nll(z, bva[0], bva[1]).item()
            vc = antolini_cindex(surv_from_logits(z), va.df.duration.to_numpy(), va.df.event.to_numpy())
            curve["val"].append({"epoch": ep + 1, "step": step, "val_nll": vnll, "val_c_td": vc})
            print(f"epoch {ep + 1}: val nll {vnll:.4f}  val C_td {vc:.4f}  ({time.time() - t_ep:.0f}s)", flush=True)
            if vnll < best[0]:
                best = (vnll, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()})
        if not pretrain and best[1] is not None:
            model.load_state_dict(best[1])

    if a.pretrain_epochs and a.model == "oft":
        run_epochs(a.pretrain_epochs, 5e-4, pretrain=True)
    run_epochs(a.epochs, a.lr, pretrain=False)
    torch.save({"state": model.state_dict(), "cols": cols, "mu": sc.mu, "sd": sc.sd, "args": vars(a)}, CKPT / f"{tag}.pt")

    z = predict(model, a.model, te, gte, dev)
    surv = surv_from_logits(z)
    np.save(RES / f"surv_{tag}.npy", surv.astype(np.float32))
    res = evaluate(tag, surv, te.df, tr.df)
    res["n_params"] = n_params(model)
    res["train_examples"] = len(tr.df)
    res["minutes"] = round((time.time() - t0) / 60, 1)
    print(json.dumps(res), flush=True)
    (RES / f"{tag}.json").write_text(json.dumps(res, indent=1))
    (RES / f"{tag}_curve.json").write_text(json.dumps(curve))


if __name__ == "__main__":
    main()
