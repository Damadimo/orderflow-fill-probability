"""Dataset statistics for the report: labels per split/side, censoring, fill times, quality checks.

Run:  .venv/bin/python scripts/data_stats.py   ->  results/data_stats.json, results/data_stats_days.csv
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from fillprob.data import PROC, SPLITS, WARMUP_US, combine_labels  # noqa: E402
from fillprob.metrics import km_curve  # noqa: E402

RES = ROOT / "results"
RES.mkdir(exist_ok=True)


def main():
    rows, out = [], {}
    for split, days in SPLITS.items():
        for day in days:
            st = json.loads((PROC / f"{day}_stats.json").read_text())
            lab = combine_labels(pd.read_parquet(PROC / f"{day}_labels.parquet"))
            lab = lab[lab.t - lab.t.min() >= WARMUP_US]
            secs = st["seconds_total"]
            rows.append(dict(
                split=split, day=day, labels=len(lab), labels_bid=int((lab.side == 0).sum()), labels_ask=int((lab.side == 1).sum()),
                fill_rate_300=float(lab.event.mean()), fill_rate_300_trade=float(lab.event_trade.mean()),
                cross_first=float((lab.fill_type == 3).mean()),
                censored_300=float(((lab.event == 0) & (lab.duration >= 299.999)).mean()),
                gap_censored=float(((lab.event == 0) & lab.gap_censored.astype(bool)).mean()),   # unfilled labels cut by a gap median_fill_s=float(lab[lab.event == 1].duration.median()),
                order_events_m=(st.get("n_order_created", 0) + st.get("n_order_deleted", 0) + st.get("n_order_changed", 0)) / 1e6,
                trades=st["trades"], chain_breaks=st["chain_breaks"],
                frac_seconds_dropped=(st["seconds_invalid"] + st.get("seconds_quarantine", 0) + st["seconds_crossed"]) / max(1, secs),
                chain_monitored=bool(st.get("chain_monitored", True)),
                fifo_oldest=st["fifo_oldest"] / max(1, st["fifo_checked"]),
                fifo_sweep=st["fifo_sweep_consistent"] / max(1, st["fifo_checked"]),
                unknown_maker=st["trades_unknown_maker"] / max(1, st["trades"]),
                fleeting_frac_of_cancels=st.get("n_fleeting", 0) / max(1, st.get("n_delete_cancel", 1)),
                mid=float(lab.mid.median() / 100), q_n_median=float(lab.q_n.median()), q_vol_median=float(lab.q_vol.median()),
            ))
    days = pd.DataFrame(rows)
    days.to_csv(RES / "data_stats_days.csv", index=False)
    for split, g in days.groupby("split", sort=False):
        w = g.labels / g.labels.sum()
        out[split] = dict(days=len(g), labels=int(g.labels.sum()), labels_bid=int(g.labels_bid.sum()), labels_ask=int(g.labels_ask.sum()),
                          fill_rate_300=float((g.fill_rate_300 * w).sum()), fill_rate_300_trade=float((g.fill_rate_300_trade * w).sum()),
                          cross_first=float((g.cross_first * w).sum()), censored_300=float((g.censored_300 * w).sum()),
                          days_chain_monitored=int(g.chain_monitored.sum()),
                          gap_censored=float((g.gap_censored * w).sum()), frac_seconds_dropped=float(g.frac_seconds_dropped.mean()),
                          fifo_oldest=float(g.fifo_oldest.mean()), fifo_sweep=float(g.fifo_sweep.mean()),
                          unknown_maker=float(g.unknown_maker.mean()), chain_breaks=int(g.chain_breaks.sum()),
                          order_events_m=float(g.order_events_m.sum()), trades=int(g.trades.sum()))
    # fill-time distribution on the training days
    tr = pd.concat([combine_labels(pd.read_parquet(PROC / f"{d}_labels.parquet")) for d in SPLITS["train"]])
    grid = np.array([1, 5, 10, 30, 60, 120, 300.0])
    out["train_km_fill_prob"] = {str(int(t)): float(1 - s) for t, s in zip(grid, km_curve(tr.duration.to_numpy(), tr.event.to_numpy(), grid))}
    out["train_fill_time_quantiles_filled"] = {str(q): float(v) for q, v in tr[tr.event == 1].duration.quantile([.1, .25, .5, .75, .9]).items()}
    (RES / "data_stats.json").write_text(json.dumps(out, indent=1))
    print(days.round(3).to_string())
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
