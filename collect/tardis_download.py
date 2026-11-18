"""Download historical Bitstamp order-by-order data from Tardis.dev (free first day of each month).

Tardis serves its raw replay API one minute per request, gzip-compressed, as lines of
"<ISO receive time> <raw Bitstamp websocket message>". Each day starts with a generated
full L3 REST snapshot ("event": "snapshot"). Files are saved per minute, compressed as served:
    data/tardis/bitstamp_<pair>/<YYYY-MM-DD>/<HHMM>.gz
Re-running skips minutes that are already on disk.

Run:  .venv/bin/python collect/tardis_download.py 2025-12-01 2026-11-01 --pair btcusd
"""

import argparse
import asyncio
import datetime as dt
import json
from pathlib import Path

import httpx

OUT = Path(__file__).resolve().parent.parent / "data" / "tardis"
API = "https://api.tardis.dev/v1/data-feeds/bitstamp"


def first_days(start: dt.date, end: dt.date) -> list[dt.date]:
    days, d = [], start.replace(day=1)
    while d <= end:
        days.append(d)
        d = (d.replace(day=28) + dt.timedelta(days=4)).replace(day=1)
    return days


async def fetch_minute(client: httpx.AsyncClient, sem: asyncio.Semaphore, pair: str, t: dt.datetime, path: Path) -> None:
    if path.exists() and path.stat().st_size > 0:
        return
    filters = json.dumps([{"channel": "live_orders", "symbols": [pair]}, {"channel": "live_trades", "symbols": [pair]}])
    params = {"from": t.strftime("%Y-%m-%dT%H:%M:%S.000Z"), "to": (t + dt.timedelta(minutes=1)).strftime("%Y-%m-%dT%H:%M:%S.000Z"), "filters": filters}
    for attempt in range(8):
        async with sem:
            try:
                async with client.stream("GET", API, params=params, headers={"Accept-Encoding": "gzip"}, timeout=120) as r:
                    if r.status_code == 429:
                        raise httpx.HTTPStatusError("rate limited", request=r.request, response=r)
                    r.raise_for_status()
                    body = b"".join([chunk async for chunk in r.aiter_raw()])
                tmp = path.with_suffix(".part")
                tmp.write_bytes(body)
                tmp.rename(path)
                return
            except Exception as e:  # noqa: BLE001 - retry any network/rate-limit failure
                wait = min(2**attempt, 60)
                print(f"retry {t} attempt {attempt}: {e!r}; sleeping {wait}s", flush=True)
        await asyncio.sleep(wait)
    print(f"FAILED {t}", flush=True)


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("start")
    ap.add_argument("end")
    ap.add_argument("--pair", default="btcusd")
    ap.add_argument("--concurrency", type=int, default=24)
    args = ap.parse_args()
    days = first_days(dt.date.fromisoformat(args.start), dt.date.fromisoformat(args.end))
    sem = asyncio.Semaphore(args.concurrency)
    async with httpx.AsyncClient() as client:
        for day in reversed(days):  # newest first: those days are the test split
            day_dir = OUT / f"bitstamp_{args.pair}" / day.isoformat()
            day_dir.mkdir(parents=True, exist_ok=True)
            t0 = dt.datetime.combine(day, dt.time(), tzinfo=dt.timezone.utc)
            minutes = [t0 + dt.timedelta(minutes=m) for m in range(1440)]
            await asyncio.gather(*(fetch_minute(client, sem, args.pair, t, day_dir / t.strftime("%H%M.gz")) for t in minutes))
            n = sum(1 for p in day_dir.glob("*.gz"))
            size = sum(p.stat().st_size for p in day_dir.glob("*.gz")) / 1e6
            print(f"done {day} {args.pair}: {n}/1440 minutes, {size:.0f} MB", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
