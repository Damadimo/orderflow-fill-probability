"""Evaluation metrics for predicted fill-time survival curves (numpy only).

Conventions shared by every function:
- `surv` is an [N, K] matrix of S_i(t) = P(no fill by t) on a time grid `grid` (default EDGES[1:],
  the right edges of the model's bins): one row per test order, one column per grid time. Grid
  times must be strictly increasing, and positive for surv_at. Shapes are checked, and NaN
  predictions give NaN metrics instead of a plausible-looking number.
- `durations`, `events` are the observed fill/censor times (seconds) and fill indicators.
- IPCW metrics estimate the censoring survival G(t) by Kaplan-Meier on the training labels
  (`train_durations`, `train_events`); pass None to use the test labels instead. G is a
  right-continuous step function, held constant after the last training time (scikit-survival
  raises an error there instead).
- Pair counting is exact and runs in O(N log^2 N) (merge-sort tree), so N = 200k takes well under a
  second. Antolini's C counts predictions as tied only when exactly equal; Uno's C uses
  scikit-survival's tolerance (|risk_i - risk_j| <= 1e-8 is a tie).
"""
import numpy as np
import pandas as pd

from .bins import EDGES

GRID = EDGES[1:]


def _te(durations, events):
    t = np.asarray(durations, dtype=float).ravel()
    e = np.asarray(events).astype(bool).ravel()
    if len(t) != len(e) or np.isnan(t).any():
        raise ValueError("durations and events must have the same length and durations no NaN")
    return t, e


def _grid(surv, grid, n=None):
    """surv as float [N, K] and its grid [K], after checking shapes and that the grid increases."""
    s = np.asarray(surv, dtype=float)
    g = GRID if grid is None else np.asarray(grid, dtype=float).ravel()
    if s.ndim != 2 or s.shape[1] != len(g) or (n is not None and len(s) != n):
        raise ValueError(f"surv must be [{'N' if n is None else n}, {len(g)}] (one row per order, "
                         f"one column per grid time), got {s.shape}")
    if np.any(np.diff(g) <= 0):
        raise ValueError("grid times must be strictly increasing")
    return s, g


def _km(t, e, reverse=False):
    """Kaplan-Meier at the unique times of t. reverse=True estimates the censoring survival G;
    at tied times fills are removed from the risk set before censorings (as in scikit-survival)."""
    ut, inv = np.unique(t, return_inverse=True)
    n = np.bincount(inv, minlength=len(ut)).astype(float)
    d = np.bincount(inv, weights=e.astype(float), minlength=len(ut))
    at_risk = len(t) - np.concatenate([[0.0], np.cumsum(n)[:-1]])
    if reverse:
        at_risk, d = at_risk - d, n - d
    return ut, np.cumprod(1.0 - np.divide(d, at_risk, out=np.zeros_like(d), where=d > 0))


def _step(ut, s, t):
    """Right-continuous step function through (ut, s); 1 before ut[0], s[-1] after ut[-1]."""
    return np.concatenate([[1.0], s])[np.searchsorted(ut, t, side="right")]


def _censoring(train_durations, train_events, durations, events):
    t, e = _te(durations, events) if train_durations is None else _te(train_durations, train_events)
    ut, g = _km(t, e, reverse=True)
    return lambda x: _step(ut, g, np.asarray(x, dtype=float))


def km_curve(durations, events, times):
    """Kaplan-Meier survival P(T > t) at `times`."""
    t, e = _te(durations, events)
    return _step(*_km(t, e), np.asarray(times, dtype=float))


def surv_at(surv, t, grid=None, interp="loglinear"):
    """Survival at arbitrary times t (scalar or [M]) from `surv` on `grid`; returns [N, M].

    interp="loglinear": constant hazard between grid times with S(0) = 1 (exact on the grid and
    for exponential fill times); interp="step": value at the largest grid time <= t (1 before the
    first). Both hold S constant after the last grid time.
    """
    s, g = _grid(surv, grid)
    if g[0] <= 0 or interp not in ("loglinear", "step"):
        raise ValueError("surv_at needs positive grid times and interp 'loglinear' or 'step'")
    g = np.concatenate([[0.0], g])
    s = np.hstack([np.ones((len(s), 1)), s])
    t = np.atleast_1d(np.asarray(t, dtype=float))
    k = np.searchsorted(g, t, side="right") - 1  # last grid index <= t
    if interp == "step":
        return s[:, np.clip(k, 0, len(g) - 1)]
    k = np.clip(k, 0, len(g) - 2)
    w = np.clip((t - g[k]) / (g[k + 1] - g[k]), 0.0, 1.0)
    log_s = np.log(np.maximum(s, 1e-300))
    return np.exp(log_s[:, k] * (1.0 - w) + log_s[:, k + 1] * w)


def _at(surv, times, grid, t):
    """surv_at for the test orders, checking that surv has one row per order."""
    s = surv_at(surv, times, grid)
    if len(s) != len(t):
        raise ValueError(f"surv has {len(s)} rows but there are {len(t)} orders")
    return s


def _dominance(p, v, q, w, tol=0.0):
    """For each query i, over points j with p_j > q_i: n_i = #{j}, lt_i = #{v_j < w_i - tol} and
    eq_i = #{|v_j - w_i| <= tol}. Points sorted by p descending turn {p_j > q_i} into a prefix of
    length n_i; the prefix splits into <= log2(N) aligned blocks, each answered by binary search
    in a per-level sorted key array (merge-sort tree)."""
    o = np.argsort(-p, kind="stable")
    n = len(p) - np.searchsorted(p[o][::-1], q, side="right")
    vals = np.unique(v)
    r = np.searchsorted(vals, v[o])  # rank of each point's value
    r_lo = np.searchsorted(vals, w - tol, side="left")  # point j has v_j < w_i - tol  <=>  r_j < r_lo_i
    r_hi = np.searchsorted(vals, w + tol, side="right")  # point j has v_j <= w_i + tol <=>  r_j < r_hi_i
    lt, le = np.zeros(len(q), np.int64), np.zeros(len(q), np.int64)
    pos, base = np.arange(len(p)), len(vals) + 1
    level = 0
    while (1 << level) <= len(p):
        sel = np.flatnonzero((n >> level) & 1)
        if len(sel):
            keys = np.sort((pos >> level) * base + r)  # ranks sorted within each block of 2**level
            b = (n[sel] >> level) - 1  # the block of this size inside the prefix
            lt[sel] += np.searchsorted(keys, b * base + r_lo[sel]) - (b << level)
            le[sel] += np.searchsorted(keys, b * base + r_hi[sel]) - (b << level)
        level += 1
    return n, lt, le - lt


def _time_order(t, e):
    """Integer keys with: sample j comparable to fill i  <=>  key_j > 2 * rank(t_i), i.e.
    t_j > t_i, or t_j == t_i and j censored (censored orders outlive same-time fills)."""
    rank = np.unique(t, return_inverse=True)[1].astype(np.int64)
    return 2 * rank + ~e, 2 * rank


def antolini_cindex(surv, durations, events, max_pairs=None, seed=0, grid=None, tie_credit=0.5):
    """Time-dependent concordance C^td (Antolini et al. 2005).

    Comparable pairs: i filled and t_i < t_j (or t_j == t_i with j censored). The pair is concordant
    if S_i(t_i) < S_j(t_i), with S read off the step function on `grid` (value at the largest grid
    time <= t, S = 1 before the first grid time, so fills before 0.1 s give tied pairs). Ties earn
    `tie_credit` (0.5 as in Harrell's C; pycox's 'antolini' uses 0). grid=EDGES[:-1] instead reads
    S at the right edge of the bin that contains t.
    Exact by default (about 0.2 s for N = 200k). If max_pairs is set and there are more comparable
    pairs, max_pairs of them are drawn uniformly at random with replacement (seed), which adds a
    standard error of about sqrt(C (1 - C) / max_pairs).
    """
    t, e = _te(durations, events)
    s, g = _grid(surv, grid, len(t))
    if np.isnan(s).any():
        return np.nan
    s = np.hstack([np.ones((len(t), 1)), s])  # column c: S at g[c - 1]; column 0: S = 1
    col = np.searchsorted(g, t, side="right")  # read-off column of each sample's own time
    key, qkey = _time_order(t, e)
    fills = np.flatnonzero(e)
    if max_pairs is not None:
        if max_pairs < 1:
            raise ValueError("max_pairs must be at least 1")
        order = np.argsort(key, kind="stable")
        start = np.searchsorted(key[order], qkey[fills], side="right")  # partners of fill f: order[start[f]:]
        cnt = len(t) - start
        cum, m = np.cumsum(cnt), int(max_pairs)
        if len(cum) and cum[-1] > m:
            rng, num = np.random.default_rng(seed), 0.0
            for done in range(0, m, 1 << 20):  # chunks bound the memory
                u = rng.integers(0, cum[-1], min(1 << 20, m - done))  # uniform pair id
                f = np.searchsorted(cum, u, side="right")  # its fill
                i, j = fills[f], order[start[f] + u - (cum[f] - cnt[f])]
                si, sj = s[i, col[i]], s[j, col[i]]
                num += np.count_nonzero(si < sj) + tie_credit * np.count_nonzero(si == sj)
            return num / m
    num = den = 0.0
    for c in np.unique(col[fills]):
        a = fills[col[fills] == c]
        sa = s[a, c]
        # samples in later columns have t_j >= g[c] > t_i: comparable with every fill in column c
        later = np.sort(s[col > c, c])
        lo, hi = np.searchsorted(later, sa, side="left"), np.searchsorted(later, sa, side="right")
        num += (len(later) - hi).sum() + tie_credit * (hi - lo).sum()
        den += len(later) * len(a)
        same = np.flatnonzero(col == c)  # same column: needs the time order
        n, lt, eq = _dominance(key[same], s[same, c], qkey[a], sa)
        num += (n - lt - eq).sum() + tie_credit * eq.sum()
        den += n.sum()
    return num / den if den else np.nan


def uno_cindex_at(surv, durations, events, horizon, train_durations=None, train_events=None, grid=None,
                  tied_tol=1e-8):
    """Uno's IPCW concordance truncated at `horizon`, with risk = 1 - S(horizon).

    Same estimator as sksurv.metrics.concordance_index_ipcw(train, test, risk, tau=horizon,
    tied_tol=tied_tol): fills with t_i < horizon weighted by G(t_i)^-2, risks within tied_tol count
    as ties worth 0.5. Computed exactly in O(N log^2 N) instead of O(N^2); tests check equality.
    S(horizon) uses log-linear interpolation. Returns NaN when there is no comparable pair (sksurv
    raises), and gives fills with G(t_i) = 0 weight 0 (sksurv raises).
    """
    t, e = _te(durations, events)
    risk = 1.0 - _at(surv, horizon, grid, t)[:, 0]
    if np.isnan(risk).any():
        return np.nan
    G = _censoring(train_durations, train_events, durations, events)
    key, qkey = _time_order(t, e)
    a = np.flatnonzero(e & (t < horizon))
    ga = G(t[a])
    w = np.divide(1.0, ga**2, out=np.zeros_like(ga), where=ga > 0)
    n, lt, eq = _dominance(key, risk, qkey[a], risk[a], tied_tol)
    den = (w * n).sum()
    return float((w * (lt + 0.5 * eq)).sum() / den) if den > 0 else np.nan


def _brier(s_t, t, e, times, G):
    """IPCW Brier score (Graf et al. 1999) for S at `times` ([N, M]); nan where G(time) = 0."""
    g_t = G(t)
    w_fill = np.divide(e, g_t, out=np.zeros_like(g_t), where=g_t > 0)[:, None]
    with np.errstate(divide="ignore", invalid="ignore"):
        w_alive = (t[:, None] > times) / G(times)
        return ((t[:, None] <= times) * w_fill * s_t**2 + w_alive * (1.0 - s_t) ** 2).mean(0)


def brier_at(surv, durations, events, horizon, train_durations=None, train_events=None, grid=None):
    """IPCW Brier score at `horizon` (scalar -> float, array -> array). Same estimator as
    sksurv.metrics.brier_score; horizon should lie below the largest test duration. Returns NaN
    where G(horizon) = 0 (the score is undefined there; sksurv returns only the fills' term)."""
    t, e = _te(durations, events)
    h = np.atleast_1d(np.asarray(horizon, dtype=float))
    bs = _brier(_at(surv, h, grid, t), t, e, h, _censoring(train_durations, train_events, durations, events))
    return float(bs[0]) if np.ndim(horizon) == 0 else bs


def ibs(surv, durations, events, train_durations=None, train_events=None, times=None, grid=None):
    """Integrated IPCW Brier score: trapezoid rule over `times` divided by their span.

    `times` defaults to the grid (EDGES[1:], 0.1 to 300 s). Times are clipped into the test
    follow-up range [min duration, max duration), the range sksurv.metrics.integrated_brier_score
    accepts, so with labels censored at exactly 300 s the 300 s point becomes the largest float
    below 300 (S there is taken from the 300 s column). Times where G(t) = 0 are dropped because
    the Brier score is undefined there (sksurv keeps them and drops the alive orders' term). With
    G > 0 at every time the result equals sksurv's.
    """
    t, e = _te(durations, events)
    G = _censoring(train_durations, train_events, durations, events)
    x = np.asarray(times if times is not None else GRID if grid is None else grid, dtype=float)
    x = np.unique(np.clip(x, t.min(), np.nextafter(t.max(), -np.inf)))
    x = x[G(x) > 0]
    if len(x) < 2:
        raise ValueError("need at least two evaluation times inside the follow-up range")
    bs = _brier(_at(surv, x, grid, t), t, e, x, G)
    return float(np.trapezoid(bs, x) / (x[-1] - x[0]))


def calibration(surv_at_h, durations, events, h, n_bins=10):
    """Reliability table for P(fill by h): equal-count bins of predicted fill probability.

    surv_at_h: predicted P(no fill by h) per order, i.e. surv_at(surv, h)[:, 0]. Columns: bin, pred
    (mean predicted fill probability), obs (1 - KM(h) of the orders in the bin, so censoring is
    handled), n.
    """
    t, e = _te(durations, events)
    p = 1.0 - np.asarray(surv_at_h, dtype=float).ravel()
    if len(p) != len(t):
        raise ValueError(f"surv_at_h has {len(p)} values but there are {len(t)} orders")
    order = np.argsort(p, kind="stable")
    pos_bin = np.arange(len(p)) * n_bins // max(len(p), 1)          # equal-count bins by rank
    ps = p[order]
    first = np.r_[True, ps[1:] != ps[:-1]] if len(ps) else np.zeros(0, bool)
    b = np.empty(len(p), np.int64)
    b[order] = pos_bin[first][np.cumsum(first) - 1]                  # tied predictions share one bin
    rows = [dict(bin=k, pred=p[m].mean(), obs=1.0 - float(km_curve(t[m], e[m], h)), n=int(m.sum()))
            for k in range(n_bins) if (m := b == k).any()]
    return pd.DataFrame(rows, columns=["bin", "pred", "obs", "n"])


def ece_km(surv_at_h, durations, events, h, n_bins=10):
    """Expected calibration error at h: count-weighted mean |pred - obs| over calibration() bins."""
    df = calibration(surv_at_h, durations, events, h, n_bins)
    return float(np.average((df.pred - df.obs).abs(), weights=df.n)) if len(df) else np.nan
