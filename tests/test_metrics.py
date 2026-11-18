"""Tests for fillprob.bins and fillprob.metrics on synthetic exponential fill times."""
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pytest
import torch
from lifelines import KaplanMeierFitter
from lifelines.utils import concordance_index
from sksurv.metrics import brier_score, concordance_index_ipcw, integrated_brier_score
from sksurv.util import Surv

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fillprob.bins import EDGES, HORIZON, N_BINS, discrete_nll, surv_from_logits, to_bins  # noqa: E402
from fillprob.metrics import (  # noqa: E402
    GRID, antolini_cindex, brier_at, calibration, ece_km, ibs, km_curve, surv_at, uno_cindex_at)


def make_data(n, seed, censor=True, beta=1.0, base=1 / 20):
    """PH exponential fill times: rate = base * exp(beta * x); random censoring plus the 300 s horizon."""
    rng = np.random.default_rng(seed)
    lam = base * np.exp(beta * rng.normal(size=n))
    t = rng.exponential(1 / lam)
    if not censor:
        return lam, t, np.ones(n, bool)
    c = np.minimum(rng.exponential(150.0, n), HORIZON)
    return lam, np.minimum(t, c), t <= c


def oracle(lam, times=GRID):
    return np.exp(-np.outer(lam, times))


def oracle_logits(lam):
    return np.log(np.expm1(np.outer(lam, np.diff(EDGES))))  # logit of 1 - exp(-lam * bin width)


def constant(d_tr, e_tr, n):
    return np.tile(km_curve(d_tr, e_tr, GRID), (n, 1))


# ---------------------------------------------------------------- bins.py

def test_edges_and_to_bins():
    assert len(EDGES) == N_BINS + 1 and EDGES[0] == 0 and EDGES[1] == 0.1 and EDGES[-1] == HORIZON
    d = [0.0, 0.05, 0.1, 0.1000001, EDGES[5], EDGES[5] + 1e-9, 299.9, 300.0, 300.0, 301.0]
    e = [1, 1, 1, 1, 1, 1, 1, 0, 1, 1]
    idx, ev = to_bins(d, e)
    assert idx.dtype == np.int64
    assert idx.tolist() == [0, 0, 0, 1, 4, 5, 19, 19, 19, 19]
    assert ev.tolist() == [1, 1, 1, 1, 1, 1, 1, 0, 1, 0]  # > 300 s is censored


def test_discrete_nll_matches_brute_force():  # (1)
    rng = np.random.default_rng(0)
    b = 200
    z = rng.normal(0, 3, (b, N_BINS))
    k, ev = rng.integers(0, N_BINS, b), rng.integers(0, 2, b)
    h = 1 / (1 + np.exp(-z))
    ll = [np.log(1 - h[i, :k[i]]).sum() + (np.log(h[i, k[i]]) if ev[i] else np.log(1 - h[i, k[i]]))
          for i in range(b)]
    zt = torch.tensor(z, requires_grad=True)
    loss = discrete_nll(zt, k, ev)
    assert loss.item() == pytest.approx(-np.mean(ll), abs=1e-12)
    # gradient equals autograd through the naive formula
    ht = torch.sigmoid(zt)
    naive = -torch.stack([torch.log(1 - ht[i, :k[i]]).sum()
                          + (torch.log(ht[i, k[i]]) if ev[i] else torch.log(1 - ht[i, k[i]]))
                          for i in range(b)]).mean()
    g1, = torch.autograd.grad(loss, zt)
    g2, = torch.autograd.grad(naive, zt)
    assert torch.allclose(g1, g2, atol=1e-12)
    # extreme logits stay finite in float32
    zx = torch.tensor([[80.0] * N_BINS, [-80.0] * N_BINS], requires_grad=True)
    lx = discrete_nll(zx, torch.tensor([3, 19]), torch.tensor([0, 1]))
    lx.backward()
    assert torch.isfinite(lx) and torch.isfinite(zx.grad).all()


def test_surv_from_logits_and_interpolation():
    lam = np.array([0.01, 0.1, 1.0])
    s = surv_from_logits(torch.tensor(oracle_logits(lam)))
    assert np.allclose(s, oracle(lam), rtol=1e-10, atol=1e-14)
    t = np.array([0.03, 1.0, 10.0, 60.0, 299.0])  # off-grid; log-linear is exact for exponentials
    assert np.allclose(surv_at(s, t), oracle(lam, t), rtol=1e-9, atol=1e-14)
    st = surv_at(s, [0.05, 0.1, 1.0, 1e4], interp="step")
    assert np.allclose(st[:, 0], 1) and np.allclose(st[:, 1], s[:, 0])
    assert np.allclose(st[:, 2], s[:, 5]) and np.allclose(st[:, 3], s[:, -1])  # EDGES[6] = 0.82 <= 1 < 1.25


def test_oracle_logits_have_lower_nll():
    lam, d, e = make_data(20000, 3)
    k, ev = to_bins(d, e)
    nll_or = discrete_nll(torch.tensor(oracle_logits(lam)), k, ev).item()
    s_c = km_curve(d, e, GRID)
    h_c = 1 - s_c / np.concatenate([[1.0], s_c[:-1]])
    nll_c = discrete_nll(torch.tensor(np.tile(np.log(h_c / (1 - h_c)), (len(d), 1))), k, ev).item()
    assert nll_or < nll_c - 0.05


# ---------------------------------------------------------------- metrics.py: agreement with references

def test_km_matches_lifelines():
    _, d, e = make_data(3000, 4)
    d = np.round(d, 1)  # ties
    times = np.array([0.05, 0.1, 1.0, 7.3, 30.0, 299.0, 300.0])
    ref = KaplanMeierFitter().fit(d, e).survival_function_at_times(times).to_numpy()
    assert np.allclose(km_curve(d, e, times), ref, atol=1e-12)


def brute_antolini(surv, d, e, grid=GRID, tie=0.5):
    col = np.searchsorted(grid, d, side="right")
    s = np.hstack([np.ones((len(d), 1)), surv])
    num = den = 0.0
    for i in np.flatnonzero(e):
        for j in range(len(d)):
            if d[i] < d[j] or (d[i] == d[j] and not e[j]):
                si, sj = s[i, col[i]], s[j, col[i]]
                den += 1
                num += (si < sj) + tie * (si == sj)
    return num / den


def test_antolini_exact_with_ties():
    rng = np.random.default_rng(5)
    n = 400
    lam, d, e = make_data(n, 6)
    # tied times, on-grid times, fills before 0.1 s and at/after 300 s
    d[:120] = rng.choice([0.05, 0.1, EDGES[7], 5.0, 10.0, 300.0, 400.0], 120)
    surv = np.round(oracle(lam * rng.lognormal(0, 0.5, n)), 2)  # many prediction ties
    surv[:50] = surv[50:100]
    for tie in (0.5, 0.0):
        ref = brute_antolini(surv, d, e, tie=tie)
        assert antolini_cindex(surv, d, e, tie_credit=tie) == pytest.approx(ref, abs=1e-12)
    ref = brute_antolini(surv, d, e, EDGES[:-1])
    assert antolini_cindex(surv, d, e, grid=EDGES[:-1]) == pytest.approx(ref, abs=1e-12)


def test_antolini_max_pairs_subsample():
    # max_pairs draws comparable pairs uniformly. The old version subsampled fills instead: at
    # N = 200k with the max_pairs=2e6 that evaluate.py passes, it kept about 20 fills (SD 0.06).
    lam, d, e = make_data(200_000, 7)
    s = oracle(lam)
    exact = antolini_cindex(s, d, e)
    sub = antolini_cindex(s, d, e, max_pairs=2_000_000, seed=1)
    assert sub != exact and abs(sub - exact) < 0.003  # standard error ~3e-4
    assert sub == antolini_cindex(s, d, e, max_pairs=2_000_000, seed=1)
    assert sub != antolini_cindex(s, d, e, max_pairs=2_000_000, seed=2)
    small = slice(0, 300)  # fewer comparable pairs than max_pairs: exact
    assert antolini_cindex(s[small], d[small], e[small], max_pairs=10**6) == antolini_cindex(s[small], d[small], e[small])
    s_const = np.tile(s.mean(0), (len(d), 1))  # every sampled pair tied
    assert antolini_cindex(s_const, d, e, max_pairs=10**5) == 0.5
    assert antolini_cindex(s_const, d, e, max_pairs=10**5, tie_credit=0.0) == 0.0


def test_uno_matches_sksurv():
    lam_tr, d_tr, e_tr = make_data(4000, 8)
    lam, d, e = make_data(3000, 9)
    d = np.round(d, 0)  # tied times
    surv = np.round(oracle(lam * np.random.default_rng(10).lognormal(0, 0.5, len(lam))), 2)  # prediction ties
    y_tr, y_te = Surv.from_arrays(e_tr, d_tr), Surv.from_arrays(e, d)
    for h in (1.0, 10.0, 60.0, 299.0):
        risk = 1 - surv_at(surv, h)[:, 0]
        ref = concordance_index_ipcw(y_tr, y_te, risk, tau=h)[0]
        assert uno_cindex_at(surv, d, e, h, d_tr, e_tr) == pytest.approx(ref, abs=1e-12)


def test_brier_and_ibs_match_sksurv():
    lam_tr, d_tr, e_tr = make_data(4000, 11)
    lam, d, e = make_data(3000, 12)
    surv = oracle(lam * np.random.default_rng(13).lognormal(0, 0.3, len(lam)))
    y_tr, y_te = Surv.from_arrays(e_tr, d_tr), Surv.from_arrays(e, d)
    h = np.array([0.5, 1.0, 10.0, 60.0, 200.0])
    ref = brier_score(y_tr, y_te, surv_at(surv, h), h)[1]
    assert np.allclose(brier_at(surv, d, e, h, d_tr, e_tr), ref, atol=1e-12)
    assert isinstance(brier_at(surv, d, e, 10.0, d_tr, e_tr), float)
    x = np.unique(np.clip(GRID, d.min(), np.nextafter(d.max(), -np.inf)))  # the clipping ibs() applies
    assert x[-1] < 300 and len(x) == N_BINS
    ref = integrated_brier_score(y_tr, y_te, surv_at(surv, x), x)
    assert ibs(surv, d, e, d_tr, e_tr) == pytest.approx(ref, abs=1e-12)


# ---------------------------------------------------------------- metrics.py: behaviour

def test_oracle_beats_constant():  # (2)
    _, d_tr, e_tr = make_data(20000, 14)
    lam, d, e = make_data(20000, 15)
    s_or, s_c = oracle(lam), constant(d_tr, e_tr, len(d))
    assert ibs(s_or, d, e, d_tr, e_tr) < ibs(s_c, d, e, d_tr, e_tr) - 0.005
    assert brier_at(s_or, d, e, 30.0, d_tr, e_tr) < brier_at(s_c, d, e, 30.0, d_tr, e_tr) - 0.005
    assert antolini_cindex(s_c, d, e) == pytest.approx(0.5)
    assert antolini_cindex(s_or, d, e) > 0.7
    assert uno_cindex_at(s_c, d, e, 30.0, d_tr, e_tr) == pytest.approx(0.5)
    assert uno_cindex_at(s_or, d, e, 30.0, d_tr, e_tr) > 0.7


def test_antolini_equals_harrell_for_ph_uncensored():  # (3)
    lam, d, e = make_data(5000, 16, censor=False)
    harrell = concordance_index(d, np.log(2) / lam, e)  # predicted median: higher means later fill
    assert abs(antolini_cindex(oracle(lam), d, e) - harrell) < 0.01


def test_oracle_calibration_near_diagonal():  # (4)
    _, d_tr, e_tr = make_data(50000, 17)
    lam, d, e = make_data(50000, 18)
    for h in (1.0, 10.0, 60.0):
        s_h = surv_at(oracle(lam), h)[:, 0]
        df = calibration(s_h, d, e, h)
        assert len(df) == 10 and df.n.sum() == len(d) and df.pred.is_monotonic_increasing
        assert (df.pred - df.obs).abs().max() < 0.03
        assert ece_km(s_h, d, e, h) < 0.015
        assert ece_km(s_h**2, d, e, h) > ece_km(s_h, d, e, h) + 0.02  # overconfident fills are flagged
    s_c = surv_at(constant(d_tr, e_tr, len(d)), 10.0)[:, 0]  # constant model: well calibrated on average
    assert ece_km(s_c, d, e, 10.0) < 0.02


def test_runtime_200k():  # (5)
    _, d_tr, e_tr = make_data(200_000, 19)
    lam, d, e = make_data(200_000, 20)
    s = surv_from_logits(torch.tensor(oracle_logits(lam), dtype=torch.float32))  # float32 model output
    t0 = time.perf_counter()
    c = antolini_cindex(s, d, e)
    u = uno_cindex_at(s, d, e, 30.0, d_tr, e_tr)
    b = ibs(s, d, e, d_tr, e_tr)
    bs = brier_at(s, d, e, [1.0, 10.0, 60.0], d_tr, e_tr)
    cal = calibration(surv_at(s, 10.0)[:, 0], d, e, 10.0)
    elapsed = time.perf_counter() - t0
    print(f"N=200k: antolini={c:.4f} uno@30={u:.4f} ibs={b:.4f} brier={np.round(bs, 4)} in {elapsed:.1f}s")
    assert elapsed < 30 and 0.7 < c < 0.8 and 0.7 < u < 0.8 and len(cal) == 10


# ---------------------------------------------------------------- edge cases (review)

def pycox_antolini(surv_tn, index, d, e):
    """Pure-python port of pycox EvalSurv(surv, d, e, steps='post').concordance_td('antolini'):
    surv_tn is [T, N] on `index`; read-off row = searchsorted(index, t, 'right') - 1, clipped at 0."""
    idx = np.clip(np.searchsorted(index, d, side="right") - 1, 0, len(index) - 1)
    num = den = 0
    for i in range(len(d)):
        for j in range(len(d)):
            comp = j != i and e[i] and (d[i] < d[j] or (d[i] == d[j] and not e[j]))
            den += comp
            num += comp and surv_tn[idx[i], i] < surv_tn[idx[i], j]
    return num / den


def test_antolini_matches_pycox_antolini():
    rng = np.random.default_rng(21)
    n = 150
    lam, d, e = make_data(n, 22)
    d = np.round(d, 0)  # tied times, including fills and censorings at the same time
    surv = np.round(oracle(lam * rng.lognormal(0, 0.5, n)), 1)  # tied predictions
    index = np.concatenate([[0.0], GRID])  # an S(0) = 1 row gives pycox the "S = 1 before 0.1 s" rule
    ref = pycox_antolini(np.vstack([np.ones((1, n)), surv.T]), index, d, e)
    assert antolini_cindex(surv, d, e, tie_credit=0.0) == pytest.approx(ref, abs=1e-12)


def test_antolini_hand_example():
    # A, C fill at 5 s; B is censored at 5 s; D is censored at 50 s; E fills at 0.5 s (before the grid).
    # Comparable: (A,B) (A,D) (C,B) (C,D) and E with all four (S = 1 before the grid: 4 ties).
    # S(1 s): A .3 < B .6, A .3 < D .9, C .6 = B .6 (tie), C .6 < D .9  ->  C = (3 + 5 * tie) / 8.
    s = np.array([[0.3], [0.6], [0.6], [0.9], [0.2]])
    d, e = np.array([5.0, 5.0, 5.0, 50.0, 0.5]), np.array([1, 0, 1, 0, 1])
    assert antolini_cindex(s, d, e, grid=[1.0]) == pytest.approx(5.5 / 8)
    assert antolini_cindex(s, d, e, grid=[1.0], tie_credit=0.0) == pytest.approx(3 / 8)


def test_uno_matches_sksurv_with_near_ties():
    # Confident models push risk = 1 - S(h) to within 1e-8 of 1, where sksurv counts pairs as ties.
    _, d_tr, e_tr = make_data(3000, 23)
    lam, d, e = make_data(2000, 24)
    surv = oracle(lam * 50)
    surv[:500] = surv[500:1000] * (1 + 1e-9)  # near-duplicate predictions
    y_tr, y_te = Surv.from_arrays(e_tr, d_tr), Surv.from_arrays(e, d)
    for h in (1.0, 10.0, 60.0):
        risk = 1 - surv_at(surv, h)[:, 0]
        ref = concordance_index_ipcw(y_tr, y_te, risk, tau=h)[0]
        assert uno_cindex_at(surv, d, e, h, d_tr, e_tr) == pytest.approx(ref, abs=1e-12)
        ref0 = concordance_index_ipcw(y_tr, y_te, risk, tau=h, tied_tol=0.0)[0]
        assert uno_cindex_at(surv, d, e, h, d_tr, e_tr, tied_tol=0.0) == pytest.approx(ref0, abs=1e-12)
    assert abs(ref - ref0) > 0.01  # the tolerance matters here


def test_discrete_nll_first_and_last_bin():
    z = torch.tensor([[0.4, -0.3, 1.1]], dtype=torch.float64)
    h = torch.sigmoid(z[0]).numpy()

    def nll(k, ev):
        return discrete_nll(z, np.array([k]), np.array([ev])).item()
    assert nll(0, 1) == pytest.approx(-np.log(h[0]))  # fill in bin 0: only h_0
    assert nll(0, 0) == pytest.approx(-np.log(1 - h[0]))  # censored in bin 0: survived bin 0
    assert nll(2, 1) == pytest.approx(-np.log((1 - h[0]) * (1 - h[1]) * h[2]))
    assert nll(2, 0) == pytest.approx(-np.log((1 - h[0]) * (1 - h[1]) * (1 - h[2])))


def test_all_censored_and_no_censoring():
    lam, d, _ = make_data(500, 25)
    s = oracle(lam)
    s30 = surv_at(s, 30.0)[:, 0]
    none, every = np.zeros(len(d), bool), np.ones(len(d), bool)
    # no 0/0 anywhere; np.errstate because lifelines' KaplanMeierFitter.fit (run by an earlier test)
    # calls np.seterr(divide="ignore", invalid="ignore") globally, which hides numpy warnings
    with warnings.catch_warnings(), np.errstate(divide="raise", invalid="raise"):
        warnings.simplefilter("error")
        assert np.isnan(antolini_cindex(s, d, none)) and np.isnan(antolini_cindex(s, d, none, max_pairs=10))
        assert np.isnan(uno_cindex_at(s, d, none, 30.0))
        assert np.allclose(km_curve(d, none, GRID), 1.0)
        assert (calibration(s30, d, none, 30.0).obs == 0).all()
        # all censored: G(30) = share still at risk, so only the alive orders' term remains
        assert brier_at(s, d, none, 30.0) == pytest.approx(np.mean((1 - s30[d > 30]) ** 2), abs=1e-12)
        # no censoring: G = 1 and the Brier score is the plain squared error
        assert brier_at(s, d, every, 30.0) == pytest.approx(np.mean(np.where(d <= 30, s30, 1 - s30) ** 2), abs=1e-12)


def test_single_sample_and_single_bin():
    assert np.isnan(antolini_cindex(oracle(np.array([0.1])), [5.0], [1]))  # N = 1: no pair
    rng = np.random.default_rng(26)
    z = torch.tensor(rng.normal(0, 2, (50, 1)))  # one-bin model
    ev = rng.integers(0, 2, 50)
    hz = torch.sigmoid(z[:, 0])
    ref = -torch.where(torch.tensor(ev).bool(), torch.log(hz), torch.log(1 - hz)).mean()
    assert discrete_nll(z, np.zeros(50, int), ev).item() == pytest.approx(ref.item(), abs=1e-12)
    assert np.allclose(surv_from_logits(z), torch.sigmoid(-z).numpy())
    s1 = np.array([[0.5], [0.8]])  # one grid time
    assert np.allclose(surv_at(s1, [0.0, 5.0, 10.0, 20.0], grid=[10.0]),
                       [[1, 0.5**0.5, 0.5, 0.5], [1, 0.8**0.5, 0.8, 0.8]])
    s4, d4, e4 = rng.random((4, 1)), np.array([3.0, 12.0, 15.0, 11.0]), np.array([1, 1, 0, 1])
    assert antolini_cindex(s4, d4, e4, grid=[10.0]) == pytest.approx(brute_antolini(s4, d4, e4, np.array([10.0])))
    lam, d, e = make_data(1000, 27)  # one calibration bin: mean prediction against 1 - KM(h)
    s_h = surv_at(oracle(lam), 10.0)[:, 0]
    df = calibration(s_h, d, e, 10.0, n_bins=1)
    assert len(df) == 1 and df.n[0] == 1000
    assert df.pred[0] == pytest.approx(1 - s_h.mean()) and df.obs[0] == pytest.approx(1 - km_curve(d, e, 10.0))
    assert calibration(s_h[:3], d[:3], e[:3], 10.0).n.tolist() == [1, 1, 1]  # empty bins are skipped


def test_ibs_drops_times_where_censoring_survival_is_zero():
    _, d_tr, e_tr = make_data(3000, 29)
    keep = d_tr < 100
    d_tr, e_tr = np.append(d_tr[keep], 100.0), np.append(e_tr[keep], False)  # G = 0 from 100 s on
    lam, d, e = make_data(2000, 30)
    s = oracle(lam)
    assert np.isnan(brier_at(s, d, e, 150.0, d_tr, e_tr)) and np.isfinite(brier_at(s, d, e, 90.0, d_tr, e_tr))
    x = GRID[GRID < 100]
    ref = integrated_brier_score(Surv.from_arrays(e_tr, d_tr), Surv.from_arrays(e, d), surv_at(s, x), x)
    assert ibs(s, d, e, d_tr, e_tr) == pytest.approx(ref, abs=1e-12)


def test_bad_inputs_raise_or_give_nan():
    lam, d, e = make_data(400, 28)
    s = oracle(lam)
    s_nan = s.copy()
    s_nan[7, 3:] = np.nan  # a diverged model must not get a plausible C-index
    assert np.isnan(antolini_cindex(s_nan, d, e)) and np.isnan(antolini_cindex(s_nan, d, e, max_pairs=100))
    assert np.isnan(uno_cindex_at(s_nan, d, e, 30.0))
    assert np.isnan(brier_at(s_nan, d, e, 30.0)) and np.isnan(ibs(s_nan, d, e))
    assert np.isnan(ece_km(surv_at(s_nan, 30.0)[:, 0], d, e, 30.0))
    for bad in (lambda: surv_at(np.hstack([s, s]), 30.0),  # 40 columns on the 20-point grid
                lambda: surv_at(s, 1.0, interp="linear"),
                lambda: antolini_cindex(s[:-1], d, e),  # one row short
                lambda: antolini_cindex(s, d, e, grid=GRID[::-1]),
                lambda: uno_cindex_at(np.vstack([s, s]), d, e, 30.0),
                lambda: brier_at(s, d[:-1], e[:-1], 30.0),
                lambda: calibration(s[:, 0], d[:-1], e[:-1], 1.0),
                lambda: to_bins([1.0, np.nan], [1, 0])):
        with pytest.raises(ValueError):
            bad()
