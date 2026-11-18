"""Common evaluation for every model: survival curves S at EDGES[1:] -> metrics dict."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .bins import EDGES, N_BINS, to_bins
from .metrics import antolini_cindex, brier_at, calibration, ece_km, ibs, surv_at, uno_cindex_at


def nll_from_surv(surv: np.ndarray, durations, events) -> float:
    """Discrete-time NLL of survival curves given at the bin right edges (same likelihood as training)."""
    s = np.clip(surv, 1e-7, 1.0)
    s_prev = np.concatenate([np.ones((len(s), 1)), s[:, :-1]], 1)
    h = np.clip(1.0 - s / s_prev, 1e-7, 1 - 1e-7)
    k, e = to_bins(durations, events)
    rows = np.arange(len(s))
    ll_event = np.log(s_prev[rows, k]) + np.log(h[rows, k])
    ll_cens = np.log(s[rows, k])
    return float(-np.mean(np.where(e == 1, ll_event, ll_cens)))


def evaluate(name: str, surv: np.ndarray, test: pd.DataFrame, train: pd.DataFrame, max_pairs=None) -> dict:
    d, e = test.duration.to_numpy(), test.event.to_numpy()
    td, te = train.duration.to_numpy(), train.event.to_numpy()
    grid = EDGES[1:]
    out = {"model": name, "n_test": len(test)}
    out["nll"] = nll_from_surv(surv, d, e)
    out["c_td"] = antolini_cindex(surv, d, e, grid=grid, max_pairs=max_pairs)
    for h in (5.0, 30.0, 120.0):
        out[f"uno_c@{int(h)}s"] = uno_cindex_at(surv, d, e, h, td, te, grid=grid)
        out[f"brier@{int(h)}s"] = float(brier_at(surv, d, e, h, td, te, grid=grid))
    out["ibs"] = ibs(surv, d, e, td, te, grid=grid)
    for h in (5.0, 30.0, 120.0):
        out[f"ece@{int(h)}s"] = ece_km(surv_at(surv, h, grid=grid)[:, 0], d, e, h)
    return out


def calibration_table(surv: np.ndarray, test: pd.DataFrame, h: float, n_bins: int = 10) -> pd.DataFrame:
    """Reliability table at h, reading S(h) log-linearly between bin edges (exact for constant in-bin hazard)."""
    return calibration(surv_at(surv, h, grid=EDGES[1:])[:, 0], test.duration.to_numpy(), test.event.to_numpy(), h, n_bins=n_bins)


assert N_BINS == 20
