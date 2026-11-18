"""Figures for the progress report (vector PDFs in report/figs/).

Run:  .venv/bin/python scripts/make_figures.py
"""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from fillprob.bins import EDGES  # noqa: E402
from fillprob.data import PROC, SPLITS, WARMUP_US, combine_labels  # noqa: E402
from fillprob.metrics import km_curve, surv_at  # noqa: E402

RES, FIGS = ROOT / "results", ROOT / "report" / "figs"
FIGS.mkdir(parents=True, exist_ok=True)
BLUE, ORANGE, AQUA, INK, INK2, GRID = "#2a78d6", "#eb6834", "#1baf7a", "#0b0b0b", "#52514e", "#e6e6e3"
plt.rcParams.update({
    "font.size": 7.5, "axes.titlesize": 8, "axes.labelsize": 7.5, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "legend.fontsize": 7, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "lines.linewidth": 1.4, "font.family": "sans-serif", "pdf.fonttype": 42, "legend.frameon": False,
})
TYPE_NAMES = ["create", "cancel", "fill", "partial fill", "amend", "trade"]


def fig_sample(day="2026-11-01", seed=3):
    """One cleaned training example: event tokens (time vs price) and the queue ahead of V."""
    lab = combine_labels(pd.read_parquet(PROC / f"{day}_labels.parquet"))
    lab["row"] = np.arange(len(lab))
    ev = np.load(PROC / f"{day}_events.npy", mmap_mode="r")
    qu = np.load(PROC / f"{day}_queue.npy", mmap_mode="r")
    cand = lab[(lab.q_n >= 5) & (lab.q_n <= 7) & (lab.event == 1) & (lab.duration.between(20, 90)) & (lab.side == 0)]
    r = cand.sample(1, random_state=seed).iloc[0]
    x = np.asarray(ev[int(r.row)], dtype=np.float32)
    q = np.asarray(qu[int(r.row)], dtype=np.float32)
    x = x[x[:, 0] > 0]
    dt = -np.expm1(x[:, 4]) / 1e3                      # seconds before s (negative)
    x[:, 2] = x[:, 2] * (r.price / 100) / 1e4          # bps from V's price -> dollars
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(5.5, 1.9), gridspec_kw={"width_ratios": [2.15, 1], "wspace": 0.25})
    groups = [("create", [1], BLUE, "o", 7), ("cancel", [2], INK2, "x", 8), ("fill / trade", [3, 4, 6], ORANGE, "^", 16), ("amend", [5], AQUA, "D", 8)]
    for name, types, c, mk, sz in groups:
        m = np.isin(x[:, 0], types)
        if m.any():
            a1.scatter(dt[m], x[m, 2], s=sz, c=c, marker=mk, lw=0.8, label=name, zorder=3, alpha=0.9)
    inq = x[:, 5] > 0
    a1.scatter(dt[inq], x[inq, 2], s=40, facecolors="none", edgecolors=INK, lw=0.8, label="queued ahead of V", zorder=4)
    a1.axhline(0, color=INK, lw=0.8, ls="--", zorder=2)
    a1.set_xlim(dt.min() - 0.8, 3.2)
    a1.text(0.2, 0.8, "V's price", fontsize=6.5, color=INK, ha="left", va="bottom")
    a1.set_xlabel("seconds before the order is posted")
    a1.set_ylabel("price \u2212 V's price ($)")
    a1.set_title(f"(a) event tokens: last {len(x)} near-touch events", loc="left")
    a1.legend(ncol=5, loc="lower left", bbox_to_anchor=(-0.02, 1.08), handletextpad=0.1, columnspacing=0.7, borderaxespad=0)
    m = q[:, 5] > 0
    sizes, ages = np.expm1(q[m, 0]) / 100, np.expm1(q[m, 1])
    left, tot = 0.0, sizes.sum()
    cmap = plt.get_cmap("Blues")
    for sz, age in zip(sizes, ages):
        a2.barh(0, sz, left=left, height=0.55, color=cmap(0.35 + 0.6 * min(1, np.log1p(age) / np.log1p(600))), edgecolor="white", lw=1.2)
        if sz > 0.14 * tot:
            a2.text(left + sz / 2, 0, f"{age:.0f}s", ha="center", va="center", color="white", fontsize=6.5)
        left += sz
    a2.barh(0, 0.025 * tot, left=left, height=0.55, color=ORANGE)
    a2.text(left + 0.04 * tot, 0, "V", va="center", fontsize=7.5, fontweight="bold", color=INK)
    a2.set_yticks([])
    a2.set_xlim(0, left * 1.15)
    a2.set_xlabel("BTC queued ahead of V")
    a2.set_title("(b) queue tokens (darker = older)", loc="left")
    a2.grid(False)
    lab_txt = f"filled after {r.duration:.0f} s" if r.event == 1 else "censored"
    a2.text(0.0, -0.75, f"{int(r.q_n)} orders, {r.q_vol:.2f} BTC ahead\nlabel: {lab_txt}", fontsize=6.6, color=INK2, va="top", transform=a2.transData)
    a2.set_ylim(-1.35, 0.6)
    fig.savefig(FIGS / "sample.pdf", bbox_inches="tight")
    plt.close(fig)
    return dict(day=day, t=int(r.t), q_n=int(r.q_n), duration=float(r.duration), event=int(r.event), n_tokens=int(len(x)))


def fig_km():
    """Kaplan-Meier fill probability by split and side."""
    grid = np.geomspace(0.1, 300, 120)
    fig, ax = plt.subplots(figsize=(2.6, 1.75))
    for (split, days), c in zip(SPLITS.items(), (BLUE, ORANGE, AQUA)):
        lab = pd.concat([combine_labels(pd.read_parquet(PROC / f"{d}_labels.parquet")) for d in days])
        lab = lab[lab.duration > 0]
        for side, ls in ((0, "-"), (1, ":")):
            g = lab[lab.side == side]
            ax.plot(grid, 1 - km_curve(g.duration.to_numpy(), g.event.to_numpy(), grid), color=c, ls=ls,
                    label=f"{split} ({'bid' if side == 0 else 'ask'})")
    ax.set_xscale("log")
    ax.set_xlabel("t (s)")
    ax.set_ylabel("P(fill by t), Kaplan-Meier")
    ax.set_xlim(0.1, 300)
    ax.legend(ncol=1, loc="upper left", handlelength=1.6)
    fig.tight_layout(pad=0.3)
    fig.savefig(FIGS / "km.pdf", bbox_inches="tight")
    plt.close(fig)


def fig_learning(tags=("oft_pre", "oft_nopre", "b4_snapshot"), names=("OFT (pretrained)", "OFT (no pretrain)", "B4 snapshot")):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(5.5, 1.6))
    for tag, name, c in zip(tags, names, (BLUE, AQUA, ORANGE)):
        p = RES / f"{tag}_curve.json"
        if not p.exists():
            continue
        cur = json.loads(p.read_text())
        tr = pd.DataFrame(cur["train"])
        va = pd.DataFrame(cur["val"])
        if len(tr):
            a1.plot(tr.epoch, tr.loss.rolling(5, min_periods=1).mean(), color=c, label=name)
        if len(va):
            a2.plot(va.epoch, va.val_nll, color=c, marker="o", ms=3, label=name)
        if tag == "oft_pre" and cur.get("pretrain"):
            pre = pd.DataFrame(cur["pretrain"])
            print("pretrain curve", pre.iloc[[0, -1]].to_dict("records"))
    a1.set_xlabel("epoch")
    a1.set_ylabel("train NLL (smoothed)")
    a1.set_title("(a) training loss", loc="left")
    a2.set_xlabel("epoch")
    a2.set_ylabel("validation NLL")
    a2.set_title("(b) validation loss", loc="left")
    a2.legend(loc="upper right")
    fig.tight_layout(pad=0.3)
    fig.savefig(FIGS / "learning.pdf", bbox_inches="tight")
    plt.close(fig)


def fig_qualitative(model_tag="oft_pre", base_tag="B2_cox_L2"):
    """Predicted vs empirical fill curves for short vs long queues, and for young vs old queues."""
    idx = pd.read_parquet(RES / "test_index.parquet")
    s_m = np.load(RES / f"surv_{model_tag}.npy")
    s_b = np.load(RES / f"surv_{base_tag}.npy")
    grid = np.geomspace(0.5, 300, 60)
    fig, axes = plt.subplots(1, 2, figsize=(5.5, 1.7), sharey=True)
    qv = idx.q_vol
    groups_a = {"short queue (q < 20th pct)": qv <= qv.quantile(0.2), "long queue (q > 80th pct)": qv >= qv.quantile(0.8)}
    mid = qv.between(qv.quantile(0.4), qv.quantile(0.6))
    age = idx.q_age_mean
    groups_b = {"young queue": mid & (age <= age[mid].quantile(0.25)), "old queue": mid & (age >= age[mid].quantile(0.75))}
    for ax, groups, title in ((axes[0], groups_a, "(a) by volume queued ahead"), (axes[1], groups_b, "(b) same volume, by queue age")):
        for (name, sel), c in zip(groups.items(), (BLUE, ORANGE)):
            sel = sel.to_numpy()
            g = idx[sel]
            ax.plot(grid, 1 - km_curve(g.duration.to_numpy(), g.event.to_numpy(), grid), color=c, lw=2.2, alpha=0.35)
            ax.plot(grid, 1 - surv_at(s_m[sel], grid).mean(0), color=c, lw=1.3, label=f"{name}")
            ax.plot(grid, 1 - surv_at(s_b[sel], grid).mean(0), color=c, lw=1.0, ls=":")
        ax.set_xscale("log")
        ax.set_xlabel("t (s)")
        ax.set_title(title, loc="left")
        ax.legend(loc="upper left")
    axes[0].set_ylabel("P(fill by t)")
    fig.text(0.99, 0.01, "thick pale = empirical (KM);  solid = OFT;  dotted = Cox (L2)", ha="right", fontsize=6.3, color=INK2)
    fig.tight_layout(pad=0.3, rect=(0, 0.04, 1, 1))
    fig.savefig(FIGS / "qualitative.pdf", bbox_inches="tight")
    plt.close(fig)


def draw_sample(a1, a2, day="2026-11-01", seed=3):
    lab = combine_labels(pd.read_parquet(PROC / f"{day}_labels.parquet"))
    lab["row"] = np.arange(len(lab))
    ev = np.load(PROC / f"{day}_events.npy", mmap_mode="r")
    qu = np.load(PROC / f"{day}_queue.npy", mmap_mode="r")
    cand = lab[(lab.q_n >= 5) & (lab.q_n <= 7) & (lab.event == 1) & (lab.duration.between(20, 90)) & (lab.side == 0)]
    r = cand.sample(1, random_state=seed).iloc[0]
    x = np.asarray(ev[int(r.row)], dtype=np.float32)
    q = np.asarray(qu[int(r.row)], dtype=np.float32)
    x = x[x[:, 0] > 0]
    dt = -np.expm1(x[:, 4]) / 1e3
    x[:, 2] = x[:, 2] * (r.price / 100) / 1e4
    groups = [("create", [1], BLUE, "o", 6), ("cancel", [2], INK2, "x", 7), ("fill/trade", [3, 4, 6], ORANGE, "^", 14), ("amend", [5], AQUA, "D", 7)]
    for name, types, c, mk, sz in groups:
        m = np.isin(x[:, 0], types)
        if m.any():
            a1.scatter(dt[m], x[m, 2], s=sz, c=c, marker=mk, lw=0.7, label=name, zorder=3, alpha=0.9)
    inq = x[:, 5] > 0
    a1.scatter(dt[inq], x[inq, 2], s=34, facecolors="none", edgecolors=INK, lw=0.8, label="ahead of V", zorder=4)
    a1.axhline(0, color=INK, lw=0.8, ls="--", zorder=2)
    a1.set_xlim(dt.min() - 0.8, 4.5)
    a1.text(0.3, 0.8, "V's\nprice", fontsize=6.2, color=INK, ha="left", va="bottom", linespacing=0.9)
    a1.set_xlabel("seconds before posting")
    a1.set_ylabel("price \u2212 V's price ($)")
    a1.set_title(f"(a) {len(x)} event tokens", loc="left")
    a1.legend(ncol=5, loc="lower left", bbox_to_anchor=(-0.02, 1.1), handletextpad=0.05, columnspacing=0.55, borderaxespad=0, fontsize=6.5)
    m = q[:, 5] > 0
    sizes, ages = np.expm1(q[m, 0]) / 100, np.expm1(q[m, 1])
    left, tot = 0.0, sizes.sum()
    cmap = plt.get_cmap("Blues")
    for sz, age in zip(sizes, ages):
        a2.barh(0, sz, left=left, height=0.6, color=cmap(0.35 + 0.6 * min(1, np.log1p(age) / np.log1p(600))), edgecolor="white", lw=1.2)
        if sz > 0.16 * tot:
            a2.text(left + sz / 2, 0, f"{age:.0f}s", ha="center", va="center", color="white", fontsize=6.2)
        left += sz
    a2.barh(0, 0.03 * tot, left=left, height=0.6, color=ORANGE)
    a2.text(left + 0.05 * tot, 0, "V", va="center", fontsize=7.5, fontweight="bold", color=INK)
    a2.set_yticks([])
    a2.set_xlim(0, left * 1.2)
    a2.set_xlabel("BTC queued ahead")
    a2.set_title("(b) queue (dark = old)", loc="left")
    a2.grid(False)
    lab_txt = f"filled after {r.duration:.0f} s" if r.event == 1 else "censored"
    a2.text(0.0, -0.62, f"{int(r.q_n)} orders ahead\n{lab_txt}", fontsize=6.3, color=INK2, va="top", transform=a2.transData)
    a2.set_ylim(-1.45, 0.6)
    return dict(day=day, t=int(r.t), q_n=int(r.q_n), duration=float(r.duration), event=int(r.event), n_tokens=int(len(x)))


def draw_km(ax):
    grid = np.geomspace(0.1, 300, 120)
    for (split, days), c in zip(SPLITS.items(), (BLUE, ORANGE, AQUA)):
        lab = pd.concat([combine_labels(pd.read_parquet(PROC / f"{d}_labels.parquet")) for d in days])
        lab = lab[lab.duration > 0]
        ax.plot(grid, 1 - km_curve(lab.duration.to_numpy(), lab.event.to_numpy(), grid), color=c, label=split)
    ax.set_xscale("log")
    ax.set_xlabel("t (s)")
    ax.set_ylabel("P(fill by t)")
    ax.set_xlim(0.1, 300)
    ax.set_xticks([0.1, 1, 10, 100])
    ax.set_xticklabels(["0.1", "1", "10", "100"])
    ax.set_ylim(0, 0.95)
    ax.set_title("(c) fill curves (KM)", loc="left")
    ax.legend(loc="upper left")


def fig_data():
    fig = plt.figure(figsize=(5.5, 1.75))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.7, 0.9, 1.15], wspace=0.55)
    a1, a2, a3 = fig.add_subplot(gs[0]), fig.add_subplot(gs[1]), fig.add_subplot(gs[2])
    info = draw_sample(a1, a2)
    draw_km(a3)
    fig.savefig(FIGS / "data.pdf", bbox_inches="tight")
    plt.close(fig)
    return info


def fig_results(model_tag="oft_pre", base_tag="B2_cox_L2", cox_val_nll=None):
    """(a) validation NLL per epoch; (b, c) predicted vs empirical fill curves on the test days."""
    fig = plt.figure(figsize=(5.5, 1.75))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.15, 1.15], wspace=0.42)
    a0, a1, a2 = fig.add_subplot(gs[0]), fig.add_subplot(gs[1]), fig.add_subplot(gs[2])
    for tag, name, c in (("oft_pre", "OFT pretrained", BLUE), ("oft_nopre", "OFT", AQUA), ("b4_snapshot", "B4 snapshot", ORANGE)):
        pth = RES / f"{tag}_curve.json"
        if pth.exists():
            va = pd.DataFrame(json.loads(pth.read_text())["val"])
            if len(va):
                a0.plot(va.epoch, va.val_nll, color=c, marker="o", ms=3, label=name)
    if cox_val_nll is not None:
        a0.axhline(cox_val_nll, color=INK2, ls="--", lw=0.9)
        a0.text(1.0, cox_val_nll, "Cox (B2+L3)", fontsize=6.2, color=INK2, va="bottom")
    a0.set_xlabel("epoch")
    a0.set_ylabel("validation NLL")
    a0.set_title("(a) learning curves", loc="left")
    a0.legend(loc="upper right", fontsize=6.3)
    idx = pd.read_parquet(RES / "test_index.parquet")
    s_m = np.load(RES / f"surv_{model_tag}.npy")
    s_b = np.load(RES / f"surv_{base_tag}.npy")
    grid = np.geomspace(0.5, 300, 60)
    qv = idx.q_vol
    mid = qv.between(qv.quantile(0.4), qv.quantile(0.6))
    age = idx.q_age_mean
    panels = ((a1, {"short queue": qv <= qv.quantile(0.2), "long queue": qv >= qv.quantile(0.8)}, "(b) by volume ahead"),
              (a2, {"young queue": mid & (age <= age[mid].quantile(0.25)), "old queue": mid & (age >= age[mid].quantile(0.75))}, "(c) same volume, by age"))
    for ax, groups, title in panels:
        for (name, sel), c in zip(groups.items(), (BLUE, ORANGE)):
            sel = sel.to_numpy()
            g = idx[sel]
            ax.plot(grid, 1 - km_curve(g.duration.to_numpy(), g.event.to_numpy(), grid), color=c, lw=3.0, alpha=0.3)
            ax.plot(grid, 1 - surv_at(s_m[sel], grid).mean(0), color=c, lw=1.3, label=name)
            ax.plot(grid, 1 - surv_at(s_b[sel], grid).mean(0), color=c, lw=1.0, ls=":")
        ax.set_xscale("log")
        ax.set_xticks([1, 10, 100])
        ax.set_xticklabels(["1", "10", "100"])
        ax.set_xlabel("t (s)")
        ax.set_ylim(0, 1)
        ax.set_title(title, loc="left")
        ax.legend(loc="upper left", fontsize=6.3)
    a1.set_ylabel("P(fill by t)")
    fig.savefig(FIGS / "results.pdf", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    which = sys.argv[1:] or ["data", "learning", "qualitative"]
    if "data" in which:
        print(fig_data())
    if "results" in which:
        fig_results(*(sys.argv[2:4] if len(sys.argv) > 3 else ()))
    if "sample" in which:
        print(fig_sample())
    if "km" in which:
        fig_km()
    if "learning" in which:
        fig_learning()
    if "qualitative" in which:
        fig_qualitative()
