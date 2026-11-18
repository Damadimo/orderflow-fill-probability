"""Recompute ECE@5/30/120 s for every saved test survival array after the tie-binning fix in metrics.calibration.

Run:  .venv/bin/python scripts/recompute_ece.py   (patches results/baselines.json and results/<tag>.json in place)
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from fillprob.bins import EDGES  # noqa: E402
from fillprob.metrics import ece_km, surv_at  # noqa: E402

RES = ROOT / "results"


def ece(surv, d, e):
    return {f"ece@{int(h)}s": ece_km(surv_at(surv, h, grid=EDGES[1:])[:, 0], d, e, h) for h in (5.0, 30.0, 120.0)}


def main():
    idx = pd.read_parquet(RES / "test_index.parquet")
    d, e = idx.duration.to_numpy(), idx.event.to_numpy()
    base = json.loads((RES / "baselines.json").read_text())
    for r in base:
        p = RES / f"surv_{r['model']}.npy"
        if p.exists():
            new = ece(np.load(p), d, e)
            print(r["model"], {k: (round(r[k], 4), round(v, 4)) for k, v in new.items()})
            r.update(new)
    (RES / "baselines.json").write_text(json.dumps(base, indent=1))
    for p in sorted(RES.glob("surv_*.npy")):
        tag = p.stem[5:]
        js = RES / f"{tag}.json"
        if not js.exists() or tag.startswith("smoke"):
            continue
        r = json.loads(js.read_text())
        new = ece(np.load(p), d, e)
        print(tag, {k: (round(r[k], 4), round(v, 4)) for k, v in new.items()})
        r.update(new)
        js.write_text(json.dumps(r))


if __name__ == "__main__":
    main()
