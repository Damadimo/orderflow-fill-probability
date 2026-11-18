"""Baselines B1-B3 on the held-out test days.

B1 Kaplan-Meier: empirical fill-time curve per side (no book information).
B2 Cox model on L2-observable features; B2+L3 adds the queue-composition summary features.
B3 queue heuristic: P(fill <= t) = share of past-hour windows of length t whose depletion volume on
   V's side (traded against it + cancelled at the best level) exceeded the volume queued ahead of V.

Run:  .venv/bin/python scripts/baselines.py
Writes results/baselines.json and results/surv_<model>.npy (test-set survival curves).
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from fillprob.bins import EDGES  # noqa: E402
from fillprob.data import L2_FEATURES, L3_FEATURES, PROC, SPLITS, Standardizer, load_days  # noqa: E402
from fillprob.evaluate import evaluate  # noqa: E402
from fillprob.metrics import km_curve  # noqa: E402

RES = ROOT / "results"
RES.mkdir(exist_ok=True)
GRID = EDGES[1:]
TEST_STRIDE = 3


def kaplan_meier(train, test):
    surv = np.zeros((len(test), len(GRID)))
    for side in (0, 1):
        tr = train[train.side == side]
        surv[test.side.to_numpy() == side] = km_curve(tr.duration.to_numpy(), tr.event.to_numpy(), GRID)
    return surv


def cox(train, test, cols, penalizer=0.01):
    sc = Standardizer(cols).fit(train)
    xtr = pd.DataFrame(sc(train), columns=cols)
    xtr["duration"], xtr["event"] = train.duration.to_numpy(), train.event.to_numpy()
    cph = CoxPHFitter(penalizer=penalizer).fit(xtr, "duration", "event")
    xte = pd.DataFrame(sc(test), columns=cols)
    surv = cph.predict_survival_function(xte, times=GRID).to_numpy().T
    return surv, cph


def queue_heuristic(test, window_s=3600, res_s=0.1):
    """Empirical P(volume traded against V's side within t >= queue ahead), from the past hour."""
    surv = np.ones((len(test), len(GRID)))
    day_arr = test.day.to_numpy()
    for day, g in test.groupby("day"):
        pos = np.flatnonzero(day_arr == day)            # positions of g's rows in test (groupby keeps row order)
        tr = pd.read_parquet(PROC / f"{day}_trades.parquet")
        cn = pd.read_parquet(PROC / f"{day}_cancels.parquet")
        t0 = (g.t.min() // 86_400_000_000) * 86_400_000_000
        n = int(86_400 / res_s) + 1
        for side in (0, 1):
            sel = g.side.to_numpy() == side
            if not sel.any():
                continue
            # queue depletion = volume traded against V's side + volume cancelled at the best level of V's side
            ts = tr[tr.maker_side == side]
            cs = cn[cn.side == side]
            vol = np.zeros(n)
            np.add.at(vol, np.clip(((ts.t - t0) / 1e6 / res_s).astype(int), 0, n - 1), ts.amount.to_numpy())
            np.add.at(vol, np.clip(((cs.t - t0) / 1e6 / res_s).astype(int), 0, n - 1), cs.amount.to_numpy())
            cum = np.concatenate([[0.0], np.cumsum(vol)])
            gg = g[sel]
            minutes = ((gg.t - t0) // 60_000_000).to_numpy()
            out = np.ones((sel.sum(), len(GRID)))
            for k, tk in enumerate(GRID):
                w = max(1, int(round(tk / res_s)))
                for m in np.unique(minutes):
                    end = int(m * 60 / res_s)                    # windows must end before the minute starts
                    starts = np.arange(max(0, end - int(window_s / res_s)), max(0, end - w), int(1 / res_s))
                    rows = np.where(minutes == m)[0]
                    if len(starts) < 30:
                        out[rows, k] = np.nan
                        continue
                    sums = np.sort(cum[starts + w] - cum[starts])
                    q = gg.q_vol.to_numpy()[rows]
                    frac_ge = 1.0 - np.searchsorted(sums, q, side="left") / len(sums)
                    out[rows, k] = 1.0 - frac_ge
            surv[pos[sel]] = out
    # monotone non-increasing in t, and fill missing early-day minutes with the column mean
    surv = np.minimum.accumulate(np.where(np.isnan(surv), np.nanmean(surv, 0), surv), axis=1)
    return np.clip(surv, 1e-4, 1.0)


def main():
    t = time.time()
    train = load_days(SPLITS["train"], stride=4, with_tokens=False).df
    test = load_days(SPLITS["test"], stride=TEST_STRIDE, with_tokens=False).df
    print(f"train {len(train)}  test {len(test)}  load {time.time() - t:.0f}s", flush=True)
    results, survs = [], {}
    survs["B1_kaplan_meier"] = kaplan_meier(train, test)
    survs["B2_cox_L2"], cph = cox(train, test, L2_FEATURES)
    print(cph.summary[["coef", "exp(coef)", "p"]].round(3), flush=True)
    survs["B2_cox_L2_L3"], cph3 = cox(train, test, L2_FEATURES + L3_FEATURES)
    print(cph3.summary[["coef", "exp(coef)", "p"]].round(3), flush=True)
    survs["B3_queue_heuristic"] = queue_heuristic(test)
    for name, s in survs.items():
        r = evaluate(name, s, test, train)
        results.append(r)
        np.save(RES / f"surv_{name}.npy", s.astype(np.float32))
        print(json.dumps(r), flush=True)
    test[["day", "t", "side", "duration", "event", "q_vol", "q_n", "q_age_mean"]].to_parquet(RES / "test_index.parquet")
    (RES / "baselines.json").write_text(json.dumps(results, indent=1))
    print(f"done in {time.time() - t:.0f}s")


if __name__ == "__main__":
    main()
