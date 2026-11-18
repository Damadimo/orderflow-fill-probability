"""Record public order-by-order (L3) crypto feeds to hourly gzip JSON-lines files.

Each line is {"t": <local receive time, ns>, "m": <raw message text>}.
Bitstamp: live_orders + live_trades websocket channels, plus full REST order
book snapshots with order IDs (group=2) after every (re)connect and every 10 min.
Bitfinex: raw order-level book (prec R0, len 250, checksums + timestamps) and trades.
Every connect, disconnect and error is appended to data/raw/outages.log.

Run:  .venv/bin/python collect/record.py
"""

import asyncio
import gzip
import json
import time
from pathlib import Path

import httpx
import websockets

RAW = Path(__file__).resolve().parent.parent / "data" / "raw"
PAIRS = ["btcusd", "ethusd"]
SNAPSHOT_EVERY_S = 600


class HourlyWriter:
    """Append-only gzip writer that rolls over to a new file every UTC hour."""

    def __init__(self, stream: str):
        self.stream = stream
        self.hour = None
        self.fh = None

    def write(self, t_ns: int, msg: str) -> None:
        hour = time.strftime("%Y%m%d-%H", time.gmtime(t_ns / 1e9))
        if hour != self.hour:
            if self.fh:
                self.fh.close()
            day_dir = RAW / self.stream / hour[:8]
            day_dir.mkdir(parents=True, exist_ok=True)
            self.fh = gzip.open(day_dir / f"{hour}.jsonl.gz", "at", compresslevel=6)
            self.hour = hour
        self.fh.write(json.dumps({"t": t_ns, "m": msg}) + "\n")

    def flush(self) -> None:
        if self.fh:
            self.fh.flush()


def log(event: str, **kw) -> None:
    RAW.mkdir(parents=True, exist_ok=True)
    line = json.dumps({"t": time.time_ns(), "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "event": event, **kw})
    with open(RAW / "outages.log", "a") as f:
        f.write(line + "\n")
    print(line, flush=True)


async def bitstamp_snapshot(client: httpx.AsyncClient, pair: str, writer: HourlyWriter, reason: str) -> None:
    t0 = time.time_ns()
    try:
        r = await client.get(f"https://www.bitstamp.net/api/v2/order_book/{pair}/", params={"group": 2}, timeout=30)
        r.raise_for_status()
        writer.write(time.time_ns(), json.dumps({"pair": pair, "reason": reason, "t_request": t0, "book": r.json()}))
        writer.flush()
    except Exception as e:  # noqa: BLE001 - keep the recorder alive on any REST failure
        log("bitstamp_snapshot_error", pair=pair, error=repr(e))


async def bitstamp_snapshot_loop(client, writers, connected: asyncio.Event) -> None:
    while True:
        await asyncio.sleep(SNAPSHOT_EVERY_S)
        if connected.is_set():
            for pair in PAIRS:
                await bitstamp_snapshot(client, pair, writers[f"bitstamp_snapshot_{pair}"], "periodic")


async def bitstamp(writers) -> None:
    connected = asyncio.Event()
    async with httpx.AsyncClient() as client:
        asyncio.create_task(bitstamp_snapshot_loop(client, writers, connected))
        backoff = 1
        while True:
            try:
                async with websockets.connect("wss://ws.bitstamp.net", max_size=2**24, ping_interval=20, ping_timeout=20) as ws:
                    for pair in PAIRS:
                        for ch in ("live_orders", "live_trades"):
                            await ws.send(json.dumps({"event": "bts:subscribe", "data": {"channel": f"{ch}_{pair}"}}))
                    log("bitstamp_connected")
                    connected.set()
                    backoff = 1
                    # snapshots after subscribing, so the event stream already covers the snapshot time
                    for pair in PAIRS:
                        asyncio.create_task(bitstamp_snapshot(client, pair, writers[f"bitstamp_snapshot_{pair}"], "connect"))
                    last_flush = time.time()
                    async for msg in ws:
                        t = time.time_ns()
                        if '"bts:request_reconnect"' in msg:
                            log("bitstamp_request_reconnect")
                            break
                        ch = msg[msg.find('"channel"'):msg.find('"channel"') + 40]
                        pair = "btcusd" if "btcusd" in ch else "ethusd" if "ethusd" in ch else None
                        if pair is None:
                            writers["bitstamp_control"].write(t, msg)
                        else:
                            writers[f"bitstamp_{pair}"].write(t, msg)
                        if time.time() - last_flush > 5:
                            for w in writers.values():
                                w.flush()
                            last_flush = time.time()
            except Exception as e:  # noqa: BLE001
                log("bitstamp_disconnected", error=repr(e))
            connected.clear()
            log("bitstamp_reconnecting", backoff_s=backoff)
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, 60)


async def bitfinex(writers) -> None:
    backoff = 1
    while True:
        try:
            async with websockets.connect("wss://api-pub.bitfinex.com/ws/2", max_size=2**24, ping_interval=20, ping_timeout=20) as ws:
                # 131072 = checksums, 32768 = message timestamps
                await ws.send(json.dumps({"event": "conf", "flags": 131072 + 32768}))
                for sym in ("tBTCUSD", "tETHUSD"):
                    await ws.send(json.dumps({"event": "subscribe", "channel": "book", "symbol": sym, "prec": "R0", "len": "250"}))
                    await ws.send(json.dumps({"event": "subscribe", "channel": "trades", "symbol": sym}))
                log("bitfinex_connected")
                backoff = 1
                last_flush = time.time()
                async for msg in ws:
                    t = time.time_ns()
                    if msg.startswith("{") and ('"code":20051' in msg or '"code": 20051' in msg):
                        log("bitfinex_request_reconnect")
                        break
                    writers["bitfinex"].write(t, msg)
                    if time.time() - last_flush > 5:
                        writers["bitfinex"].flush()
                        last_flush = time.time()
        except Exception as e:  # noqa: BLE001
            log("bitfinex_disconnected", error=repr(e))
        log("bitfinex_reconnecting", backoff_s=backoff)
        await asyncio.sleep(backoff)
        backoff = min(backoff * 2, 60)


async def main() -> None:
    names = [f"bitstamp_{p}" for p in PAIRS] + [f"bitstamp_snapshot_{p}" for p in PAIRS] + ["bitstamp_control", "bitfinex"]
    writers = {n: HourlyWriter(n) for n in names}
    log("recorder_start")
    try:
        await asyncio.gather(bitstamp(writers), bitfinex(writers))
    finally:
        for w in writers.values():
            if w.fh:
                w.fh.close()
        log("recorder_stop")


if __name__ == "__main__":
    asyncio.run(main())
