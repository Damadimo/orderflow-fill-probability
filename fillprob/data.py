"""Assemble train/val/test arrays from the per-day replay outputs.

Splits are by calendar day (each day is a separate first-of-month sample, so no window
crosses a split boundary). Examples from the first 5 minutes of a day are dropped (the
event ring buffer is still filling). Training examples are subsampled with a stride in
seconds to cut the strong overlap between consecutive seconds.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

PROC = Path(__file__).resolve().parent.parent / "data" / "processed"

SPLITS = {
    # Bitstamp's BTC/USD tick changed from $1 to $0.01 in June 2026; val/test are 1-cent days, so the
    # three 1-cent days before them are kept in training
    "train": ["2025-12-01", "2026-01-01", "2026-02-01", "2026-03-01", "2026-04-01", "2026-05-01", "2026-06-01",
              "2026-07-01", "2026-08-01"],
    "val": ["2026-09-01"],
    "test": ["2026-10-01", "2026-11-01"],
}
WARMUP_US = 5 * 60 * 1_000_000

# hand-built features: L2-observable (Cox baseline B2) and L3 queue composition (ablation B2+L3)
L2_FEATURES = ["log_q_vol", "log_opp_vol", "imbalance", "log_vol60", "log_tv_side_60", "log_tv_at_p_60",
               "log_n_trades_60", "log_cancels_at_p_10", "log_qvol_over_tv", "log_spread_bps", "side", "hour_sin", "hour_cos"]
L3_FEATURES = ["log_q_n", "log_q_age_mean", "log_q_age_max", "q_frac_old60", "log_fleeting_10"]
GLOBAL_FEATURES = L2_FEATURES + L3_FEATURES


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["log_q_vol"] = np.log(df.q_vol + 1e-4)
    df["log_opp_vol"] = np.log(df.opp_vol + 1e-4)
    df["log_vol60"] = np.log1p(df.vol60_bps)
    df["log_tv_side_60"] = np.log1p(df.tv_side_60 * 100)
    df["log_tv_at_p_60"] = np.log1p(df.tv_at_p_60 * 100)
    df["log_n_trades_60"] = np.log1p(df.n_trades_60)
    df["log_cancels_at_p_10"] = np.log1p(df.cancels_at_p_10)
    df["log_qvol_over_tv"] = np.log((df.q_vol + 1e-4) / (df.tv_side_60 / 60 + 1e-4))
    df["log_q_n"] = np.log1p(df.q_n)
    df["log_q_age_mean"] = np.log1p(df.q_age_mean)
    df["log_q_age_max"] = np.log1p(df.q_age_max)
    df["log_fleeting_10"] = np.log1p(df.fleeting_10)
    df["log_spread_bps"] = np.log(df.spread_bps.clip(lower=1e-4))
    df["hour_sin"] = np.sin(2 * np.pi * df.hour / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df.hour / 24)
    return df


@dataclass
class Split:
    df: pd.DataFrame            # labels + features (one row per example)
    events: np.ndarray          # [N, K, 8] float16
    queue: np.ndarray           # [N, Q, 6] float16
    snaps: np.ndarray           # [N, T, 40] float16 (L2 history for the snapshot baseline)


def combine_labels(lab: pd.DataFrame) -> pd.DataFrame:
    """Primary label: V fills at the earlier of (i) a trade reaching V's queue position or printing
    through its price, and (ii) the opposite best quote moving to or through V's price (an order that
    rests there would have executed against V). The trade-only label is kept for sensitivity checks."""
    lab = lab.copy()
    lab["duration_trade"], lab["event_trade"] = lab.duration, lab.event
    cross = lab.cross_duration.notna() & (lab.cross_duration < lab.duration)
    lab["fill_type"] = np.where(cross, 3, np.where(lab.event == 1, 1, 0))     # 0 censored, 1 trade, 3 cross
    lab.loc[cross, "duration"] = lab.loc[cross, "cross_duration"]
    lab.loc[cross, "event"] = 1
    return lab


def signed_log_cents(x_cents):
    return np.sign(x_cents) * np.log1p(np.abs(x_cents))


def load_days(days: list[str], stride: int = 1, snap_hist: int = 60, with_tokens: bool = True, proc: Path = PROC,
              trade_only_labels: bool = False) -> Split:
    dfs, evs, qs, sns = [], [], [], []
    for day in days:
        lab = pd.read_parquet(proc / f"{day}_labels.parquet")
        if not trade_only_labels:
            lab = combine_labels(lab)
        lab["row"] = np.arange(len(lab))
        per_snap = lab.groupby("snap_idx")[["t", "mid"]].first()        # snap_idx is 0..n-1 (one per sampled second)
        snap_t, snap_mid = per_snap.t.to_numpy(), per_snap.mid.to_numpy()
        t0 = lab.t.min()
        keep = (lab.t - t0 >= WARMUP_US) & lab.duration.notna() & (lab.duration > 0)
        if stride > 1:
            keep &= ((lab.t - t0) // 1_000_000) % stride == 0
        lab = lab[keep]
        rows = lab.row.to_numpy()
        dfs.append(lab)
        if with_tokens:
            # event price offsets: bps from V's price -> signed log-cents (identical under $1 and 1-cent ticks)
            ev = np.load(proc / f"{day}_events.npy", mmap_mode="r")[rows].astype(np.float32)
            cents = np.round(ev[..., 2] * lab.price.to_numpy()[:, None] / 1e4)
            ev[..., 2] = signed_log_cents(cents)
            evs.append(ev.astype(np.float16))
            qs.append(np.load(proc / f"{day}_queue.npy", mmap_mode="r")[rows])
            # snapshot history (B4): offsets in signed log-cents from mid; rows older than snap_hist s are blanked
            snaps = np.load(proc / f"{day}_snaps.npy").reshape(-1, 2, 10, 2)
            snaps[..., 0] = signed_log_cents(snaps[..., 0] * snap_mid[:, None, None] / 1e4)
            snaps = snaps.reshape(len(snaps), -1)
            idx = lab.snap_idx.to_numpy()[:, None] + np.arange(-snap_hist + 1, 1)[None, :]
            ci = np.clip(idx, 0, None)
            hist = snaps[ci].astype(np.float16)
            hist[(idx < 0) | (lab.t.to_numpy()[:, None] - snap_t[ci] >= snap_hist * 1_000_000)] = 0
            # orient every snapshot relative to V: for an ask V, swap sides and flip the offset sign
            ask = lab.side.to_numpy() == 1
            h = hist[ask].reshape(-1, snap_hist, 2, 10, 2)[:, :, ::-1].copy()
            h[..., 0] *= -1
            hist[ask] = h.reshape(-1, snap_hist, 40)
            sns.append(hist)
    df = add_features(pd.concat(dfs, ignore_index=True))
    if not with_tokens:
        return Split(df, None, None, None)
    return Split(df, np.concatenate(evs), np.concatenate(qs), np.concatenate(sns))


class Standardizer:
    """Mean/std scaling fitted on training data only."""

    def __init__(self, cols):
        self.cols = cols

    def fit(self, df):
        x = df[self.cols].to_numpy(np.float64)
        self.mu, self.sd = x.mean(0), x.std(0) + 1e-6
        return self

    def __call__(self, df):
        return ((df[self.cols].to_numpy(np.float64) - self.mu) / self.sd).astype(np.float32)
