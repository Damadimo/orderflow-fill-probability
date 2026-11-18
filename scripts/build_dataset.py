"""Replay every downloaded day in parallel and write labels, tokens and per-day statistics.

Run:  .venv/bin/python scripts/build_dataset.py [--src data/tardis/bitstamp_btcusd] [--out data/processed] [--jobs 5]
Days that already have a stats file are skipped (delete it to rebuild a day).
"""

import argparse
import json
import sys
import time
from multiprocessing import Pool
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from fillprob.replay import process_day  # noqa: E402


def run(args):
    day_dir, out = args
    t = time.time()
    st = process_day(day_dir, out)
    st["seconds_to_build"] = round(time.time() - t, 1)
    (Path(out) / f"{Path(day_dir).name}_stats.json").write_text(json.dumps(st, indent=1))
    return st


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="data/tardis/bitstamp_btcusd")
    ap.add_argument("--out", default="data/processed")
    ap.add_argument("--jobs", type=int, default=5)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    days = []
    for d in sorted(Path(a.src).iterdir()):
        if not d.is_dir():
            continue
        if len(list(d.glob("*.gz"))) < 1440 and "tardis" in a.src:
            print(f"skip incomplete {d.name}")
            continue
        if not a.force and (Path(a.out) / f"{d.name}_stats.json").exists():
            continue
        days.append((str(d), a.out))
    with Pool(a.jobs) as pool:
        for st in pool.imap_unordered(run, days):
            print(st["day"], "examples", st["examples"], "chain_breaks", st["chain_breaks"], "secs", st["seconds_to_build"], flush=True)


if __name__ == "__main__":
    main()
