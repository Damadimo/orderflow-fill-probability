"""Empirical vs predicted P(fill <= 30 s) on the test days for the queue groups of Fig. 3 (b, c).

Groups (same definitions as make_figures.fig_results):
  short / long queue  = bottom / top 20% of volume queued ahead
  young / old queue   = among the middle 20% of volume ahead, bottom / top quarter of mean order age

Run:  .venv/bin/python scripts/group_stats.py   (writes results/group_stats.json)
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from fillprob.metrics import km_curve, surv_at  # noqa: E402

RES = ROOT / "results"
H = 30.0


def groups(idx: pd.DataFrame) -> dict:
    qv, age = idx.q_vol, idx.q_age_mean
    mid = qv.between(qv.quantile(0.4), qv.quantile(0.6))
    return {
        "short": (qv <= qv.quantile(0.2)).to_numpy(),
        "long": (qv >= qv.quantile(0.8)).to_numpy(),
        "young": (mid & (age <= age[mid].quantile(0.25))).to_numpy(),
        "old": (mid & (age >= age[mid].quantile(0.75))).to_numpy(),
    }


def main():
    idx = pd.read_parquet(RES / "test_index.parquet")
    g = groups(idx)
    tags = ["B1_kaplan_meier", "B2_cox_L2", "B2_cox_L2_L3", "B3_queue_heuristic", "b4_snapshot", "b4_matched", "oft_nopre", "oft_pre",
            "oft_noqueue", "oft_noevents"]
    out = {"horizon_s": H, "n": {k: int(v.sum()) for k, v in g.items()},
           "q_vol_mid_range": [float(idx.q_vol.quantile(0.4)), float(idx.q_vol.quantile(0.6))]}
    emp = {}
    for k, sel in g.items():
        d, e = idx.duration.to_numpy()[sel], idx.event.to_numpy()[sel]
        emp[k] = float(1 - km_curve(d, e, np.array([H]))[0])
        out[f"age_mean_s_{k}"] = float(idx.q_age_mean[sel].median()) if "q_age_mean" in idx else None
        out[f"q_n_median_{k}"] = float(idx.q_n[sel].median()) if "q_n" in idx else None
    out["empirical"] = emp
    for t in tags:
        p = RES / f"surv_{t}.npy"
        if not p.exists():
            continue
        s = np.load(p)
        assert len(s) == len(idx), (t, len(s), len(idx))
        f = 1 - surv_at(s, np.array([H]))[:, 0]
        out[t] = {k: float(f[sel].mean()) for k, sel in g.items()}
    print(json.dumps(out, indent=1))
    (RES / "group_stats.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
