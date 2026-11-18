"""Discrete-time survival targets and loss for limit-order fill times.

A model outputs hazard logits z_k for N_BINS log-spaced bins; h_k = sigmoid(z_k) is the
probability of a fill in bin k given no fill before it, and S(t_k) = prod_{j<=k} (1 - h_j)
is the probability of no fill by the bin's right edge t_k = EDGES[k + 1].
Bin k covers the interval (EDGES[k], EDGES[k + 1]].
"""
import numpy as np
import torch
import torch.nn.functional as F

HORIZON = 300.0  # seconds; label censoring horizon
N_BINS = 20
EDGES = np.concatenate([[0.0], np.geomspace(0.1, HORIZON, N_BINS)])  # 21 edges; EDGES[1:] are the S grid


def to_bins(durations, events):
    """Map (duration, event) to (bin index, event) for the discrete-time model.

    Durations <= 0.1 s go to bin 0. Durations > HORIZON are censored in the last bin (event 0),
    because the label only tells us the order did not fill within the horizon.
    Returns (bin_idx int64 [N], event int64 [N]). NaN durations raise (they would land in the last bin).
    """
    d = np.asarray(durations, dtype=float)
    e = np.asarray(events).astype(np.int64)
    if d.shape != e.shape or np.isnan(d).any():
        raise ValueError("durations and events must have the same shape and durations no NaN")
    idx = np.clip(np.searchsorted(EDGES, d, side="left") - 1, 0, N_BINS - 1)
    return idx.astype(np.int64), np.where(d > HORIZON, 0, e)


def discrete_nll(hazard_logits, bin_idx, events):
    """Mean negative log-likelihood of the right-censored discrete-time hazard model.

    hazard_logits: torch [B, K]; bin_idx, events: [B] (numpy or torch).
    Event in bin k:    log L = sum_{j<k} log(1 - h_j) + log h_k
    Censored in bin k: log L = sum_{j<=k} log(1 - h_j)   (the order survived the bin it was censored in)
    Uses log-sigmoid, log(1 - sigmoid(z)) = logsigmoid(-z), so large |z| stays finite.
    """
    z = hazard_logits
    idx = torch.as_tensor(bin_idx, dtype=torch.long, device=z.device).view(-1, 1)
    ev = torch.as_tensor(events, device=z.device).bool().view(-1)
    log_s = F.logsigmoid(-z).cumsum(1)  # log S through bin k
    log_s_before = F.pad(log_s[:, :-1], (1, 0))  # log S through bin k - 1 (0 for k = 0)
    ll_event = log_s_before.gather(1, idx).squeeze(1) + F.logsigmoid(z).gather(1, idx).squeeze(1)
    ll_cens = log_s.gather(1, idx).squeeze(1)
    return -torch.where(ev, ll_event, ll_cens).mean()


def surv_from_logits(logits):
    """Survival S at the N_BINS right edges (EDGES[1:]) as a float64 numpy array [B, K]."""
    z = torch.as_tensor(logits).detach().cpu().double()
    return F.logsigmoid(-z).cumsum(1).exp().numpy()
