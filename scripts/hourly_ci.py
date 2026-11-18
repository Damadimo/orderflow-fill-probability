"""Paired comparison of test C^td across the 48 test hours (Oct 1 and Nov 1), as an uncertainty estimate.

Consecutive seconds overlap, so per-order bootstraps understate the noise; hours are closer to independent.
Run:  .venv/bin/python scripts/hourly_ci.py oft_nopre B2_cox_L2_L3 [b4_matched ...]   (first tag = model of interest)
Writes results/hourly_ci.json.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from fillprob.bins import EDGES  # noqa: E402
from fillprob.metrics import antolini_cindex  # noqa: E402

RES = ROOT / "results"


def main(tags):
    idx = pd.read_parquet(RES / "test_index.parquet")
    hour = (idx.day.astype(str) + "_" + ((idx.t // 3_600_000_000) % 24).astype(str)).to_numpy()
    d, e = idx.duration.to_numpy(), idx.event.to_numpy()
    surv = {t: np.load(RES / f"surv_{t}.npy") for t in tags}
    per = {t: [] for t in tags}
    hours = sorted(set(hour))
    for h in hours:
        m = hour == h
        for t in tags:
            per[t].append(antolini_cindex(surv[t][m], d[m], e[m], grid=EDGES[1:]))
    out = {"n_hours": len(hours), "mean_c": {t: float(np.mean(v)) for t, v in per.items()}}
    a = np.array(per[tags[0]])
    for t in tags[1:]:
        diff = a - np.array(per[t])
        out[f"{tags[0]}_vs_{t}"] = {"mean_diff": float(diff.mean()), "se": float(diff.std(ddof=1) / np.sqrt(len(diff))),
                                     "hours_better": int((diff > 0).sum())}
    print(json.dumps(out, indent=1))
    (RES / "hourly_ci.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main(sys.argv[1:])
