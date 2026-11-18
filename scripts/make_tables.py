"""Generate LaTeX tables and number macros for the report from results/*.json.

Run:  .venv/bin/python scripts/make_tables.py <out_dir>   (writes tables.tex and numbers.tex)
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"


def pct(x, d=1):
    return f"{100 * x:.{d}f}"


def main():
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "report"
    out.mkdir(parents=True, exist_ok=True)
    st = json.loads((RES / "data_stats.json").read_text())
    rows = []
    for split, name in (("train", "Train (Dec--Aug)"), ("val", "Val (Sep 1)"), ("test", "Test (Oct 1, Nov 1)")):
        s = st[split]
        rows.append(f"{name} & {s['days']} & {s['labels']/1e3:,.0f}k & {pct(s['fill_rate_300'])} & {pct(s['fill_rate_300_trade'])} & "
                    f"{pct(s['censored_300'])} & {pct(s['gap_censored'])} & {pct(s['frac_seconds_dropped'])} & {pct(s['fifo_sweep'], 2)} \\\\")
    tab1 = "\n".join([
        r"\begin{tabular}{lrrrrrrrr}",
        r"\toprule",
        r"Split & Days & Labels & Filled & Filled (trades) & Cens.\ 300\,s & Cens.\ gap & Dropped & FIFO \\",
        r"\midrule", *rows, r"\bottomrule", r"\end{tabular}"])
    (out / "table_data.tex").write_text(tab1 + "\n")

    res = {}
    if (RES / "baselines.json").exists():
        for r in json.loads((RES / "baselines.json").read_text()):
            res[r["model"]] = r
    for tag in ("b4_snapshot", "oft_nopre", "oft_pre"):
        p = RES / f"{tag}.json"
        if p.exists():
            res[tag] = json.loads(p.read_text())
    names = [("B1_kaplan_meier", "B1 Kaplan--Meier"), ("B3_queue_heuristic", "B3 queue rule"), ("B2_cox_L2", "B2 Cox (L2 features)"),
             ("B2_cox_L2_L3", "B2+L3 Cox"), ("b4_snapshot", "B4 snapshot conv-Transformer"), ("oft_nopre", "OFT (no pretraining)"),
             ("oft_pre", "OFT (pretrained)")]
    best = {}
    for key, better in (("c_td", max), ("ibs", min), ("nll", min), ("brier@30s", min), ("ece@30s", min), ("uno_c@30s", max)):
        vals = [res[k][key] for k, _ in names if k in res]
        if vals:
            best[key] = better(vals)

    def fmt(k, key, d=3):
        if k not in res:
            return "--"
        v = res[k][key]
        s = f"{v:.{d}f}"
        return f"\\textbf{{{s}}}" if key in best and abs(v - best[key]) < 1e-12 else s

    rows = [f"{n} & {fmt(k, 'c_td')} & {fmt(k, 'uno_c@30s')} & {fmt(k, 'ibs')} & {fmt(k, 'brier@30s')} & {fmt(k, 'nll', 3)} & {fmt(k, 'ece@30s')} \\\\"
            for k, n in names]
    tab2 = "\n".join([
        r"\begin{tabular}{lrrrrrr}",
        r"\toprule",
        r"Model (test: Oct 1 + Nov 1) & $C^{td}\uparrow$ & $C_{30s}\uparrow$ & IBS$\downarrow$ & Brier$_{30s}\downarrow$ & NLL$\downarrow$ & ECE$_{30s}\downarrow$ \\",
        r"\midrule", *rows[:4], r"\midrule", *rows[4:], r"\bottomrule", r"\end{tabular}"])
    (out / "table_results.tex").write_text(tab2 + "\n")
    print(tab1)
    print(tab2)


if __name__ == "__main__":
    main()
