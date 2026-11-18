"""Evaluate trained models on our own live recording (never used for training or model selection).

1. Replays each recorded UTC day (data/raw/<YYYYMMDD>) with the same engine -> data/processed_live/.
2. Scores the saved checkpoints and the B1/B2 baselines (fitted on the training days) on it.

Run:  .venv/bin/python scripts/eval_new_data.py [--refresh] 20261118 20261119 ...
--refresh re-processes days that were processed while still being recorded.
Writes results/new_data.json.
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from fillprob.bins import surv_from_logits  # noqa: E402
from fillprob.data import L2_FEATURES, L3_FEATURES, SPLITS, Standardizer, load_days  # noqa: E402
from fillprob.evaluate import evaluate  # noqa: E402
from fillprob.models import OrderFlowTransformer, SnapshotConvTransformer  # noqa: E402
from fillprob.replay import process_day  # noqa: E402

LIVE = ROOT / "data" / "processed_live"
RES = ROOT / "results"


def main():
    refresh = "--refresh" in sys.argv
    days = [a for a in sys.argv[1:] if not a.startswith("--")]
    LIVE.mkdir(parents=True, exist_ok=True)
    for d in days:
        if refresh or not (LIVE / f"{d}_stats.json").exists():
            st = process_day(str(ROOT / "data" / "raw" / d), str(LIVE), recorded_pair="btcusd")
            print(d, {k: st[k] for k in ("examples", "chain_breaks", "seconds_crossed", "trades")}, flush=True)
    te = load_days(days, stride=3, proc=LIVE)
    tr = load_days(SPLITS["train"], stride=4, with_tokens=False).df
    print("new-data examples", len(te.df), flush=True)
    sys.path.insert(0, str(ROOT / "scripts"))
    from baselines import cox, kaplan_meier  # noqa: E402

    out = [evaluate("B1_kaplan_meier", kaplan_meier(tr, te.df), te.df, tr)]
    out.append(evaluate("B2_cox_L2", cox(tr, te.df, L2_FEATURES)[0], te.df, tr))
    out.append(evaluate("B2_cox_L2_L3", cox(tr, te.df, L2_FEATURES + L3_FEATURES)[0], te.df, tr))
    dev = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    for ck in sorted((ROOT / "checkpoints").glob("*.pt")):
        if ck.stem.startswith("smoke"):
            continue
        c = torch.load(ck, map_location="cpu", weights_only=False)
        sc = Standardizer(c["cols"])
        sc.mu, sc.sd = c["mu"], c["sd"]
        g = torch.from_numpy(sc(te.df))
        if c["args"]["model"] == "oft":
            m = OrderFlowTransformer(len(c["cols"]), d=c["args"]["d"], layers=c["args"]["layers"])
        else:
            m = SnapshotConvTransformer(len(c["cols"]), d=c["args"]["d"], layers=c["args"]["layers"])
        m.load_state_dict(c["state"])
        m.to(dev).eval()
        zs = []
        with torch.no_grad():
            for i in range(0, len(g), 1024):
                sl = slice(i, i + 1024)
                if c["args"]["model"] == "oft":
                    z = m(torch.from_numpy(te.events[sl].astype(np.float32)).to(dev), torch.from_numpy(te.queue[sl].astype(np.float32)).to(dev), g[sl].to(dev))
                else:
                    z = m(torch.from_numpy(te.snaps[sl].astype(np.float32)).to(dev), g[sl].to(dev))
                zs.append(z.float().cpu())
        out.append(evaluate(ck.stem, surv_from_logits(torch.cat(zs)), te.df, tr))
    for r in out:
        print(json.dumps(r), flush=True)
    hours = sum(json.loads((LIVE / f"{d}_stats.json").read_text()).get("seconds_total", 0) for d in days) / 3600
    (RES / "new_data.json").write_text(json.dumps({"days": days, "hours": hours, "n": len(te.df), "results": out}, indent=1))


if __name__ == "__main__":
    main()
