"""Replay one day of Bitstamp order-by-order (L3) messages and build fill-probability examples.

For every whole second s and each side (bid, ask) of a valid book, a virtual minimum-size
order V joins the back of the best-price queue. Its label is the time until it fills, censored
at 300 s or at a data gap. V fills at the first trade after s that either
  * prints at V's price against a resting (maker) order that was NOT already ahead of V at s
    (the incoming order has consumed everything ahead of V), or
  * prints through V's price (below a bid V / above an ask V), which price priority only
    allows once V's level is exhausted.

Model inputs saved per example:
  * hand-built features (Cox baseline and global context for the networks),
  * the last K near-touch events as tokens (relative to V),
  * the orders queued ahead of V (queue-composition tokens),
  * an index into the per-second top-10 L2 snapshot array (snapshot baseline input).

Data quality checks: the Bitstamp event_id/pre_event_id chain detects every lost message;
the book is marked invalid from a chain break until the next full snapshot. A FIFO test checks
whether each trade's maker was the oldest order at its price level.
"""

from __future__ import annotations

import gzip
import json
import math
from collections import Counter, OrderedDict, deque
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import orjson
import pandas as pd
from sortedcontainers import SortedDict

HORIZON_US = 300_000_000
SNAPSHOT_MARGIN_US = 5_000_000
QUARANTINE_US = 60_000_000   # no samples for 60 s after a lost message (chain break)
SNOWFLAKE_EPOCH_MS = 1288834974657  # Bitstamp order IDs: id >> 12 = ms since this epoch


def order_time_us(oid: int) -> int:
    """Creation time encoded in a Bitstamp order ID (verified: corr > 0.999999 with order_created times)."""
    return ((oid >> 12) + SNOWFLAKE_EPOCH_MS) * 1000
NEAR_BPS = 2.5           # events farther than this from mid are not tokenized
FLEETING_US = 500_000    # orders cancelled within 0.5 s of creation are dropped from the token stream
K_EVENTS = 256           # event tokens per example
Q_ORDERS = 32            # queue tokens per example (first orders ahead of V, in priority order)
N_LEVELS = 10            # L2 levels per side in the snapshot array
EV_CAP = 8192            # ring-buffer capacity for near-touch events

# event types for tokens
CREATE, CANCEL, FILL_DELETE, PARTIAL, AMEND, TRADE = range(6)

EVENT_FEATURES = ["type", "same_side", "px_off_bps", "log_size", "log_dt_ms", "in_queue", "log_age_ms", "taker_hits_v_side"]
QUEUE_FEATURES = ["log_size", "log_age_s", "log_rank", "log_vol_ahead", "was_partially_filled"]


@dataclass
class Order:
    side: int          # 0 bid, 1 ask
    price: int         # cents
    amount: float
    t_created: int     # exchange microseconds (snapshot orders: snapshot time, flagged)
    n_changes: int = 0
    from_snapshot: bool = False
    t_prio: int = 0    # time the order took its current queue position (creation or last amend)


@dataclass
class Virtual:
    side: int
    price: int
    t0: int
    ahead: frozenset
    row: int           # index into the output rows


@dataclass
class DayStats:
    counts: Counter = field(default_factory=Counter)
    fifo_checked: int = 0
    fifo_oldest: int = 0
    fifo_in_first3: int = 0
    trades: int = 0
    trades_unknown_maker: int = 0
    chain_breaks: int = 0
    snapshots: int = 0
    seconds_total: int = 0
    seconds_invalid: int = 0
    seconds_crossed: int = 0
    seconds_quarantine: int = 0


def iter_messages(day_dir: Path):
    """Yield parsed messages in received order from Tardis per-minute files or our recorder files."""
    files = sorted(day_dir.glob("*.gz"))
    for f in files:
        with gzip.open(f, "rb") as fh:
            for line in fh:
                if len(line) < 3:
                    continue
                if line[:1] == b"{":           # our recorder: {"t": ns, "m": raw}
                    rec = orjson.loads(line)
                    yield orjson.loads(rec["m"])
                else:                           # Tardis: "<iso time> <raw message>"
                    sp = line.find(b" ")
                    yield orjson.loads(line[sp + 1:])


class Replayer:
    def __init__(self, day: str):
        self.day = day
        self.levels = (SortedDict(), SortedDict())  # price -> {oid: True} (insertion order = priority)
        self.orders: dict[int, Order] = {}
        self.valid = False
        self.last_event_id = None
        self.stats = DayStats()
        self.clock = 0                    # max exchange time seen (us)
        self.next_sample = None
        # near-touch event ring buffer (numpy columns)
        self.ev_t = np.zeros(EV_CAP, np.int64)
        self.ev_type = np.zeros(EV_CAP, np.int8)
        self.ev_side = np.zeros(EV_CAP, np.int8)
        self.ev_px = np.zeros(EV_CAP, np.int64)
        self.ev_amt = np.zeros(EV_CAP, np.float64)
        self.ev_age = np.zeros(EV_CAP, np.int64)
        self.ev_oid = np.zeros(EV_CAP, np.int64)
        self.ev_dead = np.zeros(EV_CAP, bool)
        self.ev_n = 0
        self.create_slot: dict[int, int] = {}    # oid -> ring index of its create token
        self.fleeting_hist: deque = deque()      # times of fleeting cancels near the touch
        self.fifo_pending: list = []             # (trade t, maker, orders ahead of maker at its level)
        self.fill_time: dict[int, int] = {}      # oid -> time it was hit as maker or fully filled
        self.unknown_maker_lag_ms: list = []
        # rolling trade and cancel history for hand-built features
        self.trades_hist: deque = deque()     # (t, maker_side, price, amount)
        self.cancels_hist: deque = deque()    # (t, side, price)
        self.mid_hist: deque = deque()        # (t, mid) sampled each second
        # outputs
        self.rows: list[dict] = []
        self.ev_tokens: list[np.ndarray] = []
        self.q_tokens: list[np.ndarray] = []
        self.snap_rows: list[np.ndarray] = []  # one top-10 L2 snapshot per sampled second
        self.active: list[list[Virtual]] = [[], []]
        self.recent: deque = deque(maxlen=50_000)   # recent order messages, for snapshot alignment
        self.quarantine_until = 0
        self.last_break_t = 0
        self.vlim = [-1, 1 << 62]          # max active bid-V price, min active ask-V price
        self.crossed_prev = False
        self.cancel_rows: list = []        # (t, side, price, amount) cancels at the best level, for B3
        self.removed_prio: OrderedDict = OrderedDict()  # priority time of recently removed orders
        self.trade_rows: list = []                   # (t, maker_side, price, amount) for the trade tape
        self.replaying = False

    # ---------- book helpers ----------
    def best(self, side: int):
        lv = self.levels[side]
        if not lv:
            return None
        return lv.peekitem(-1)[0] if side == 0 else lv.peekitem(0)[0]

    def _add(self, oid: int, o: Order):
        if o.t_prio == 0:
            o.t_prio = o.t_created
        self.orders[oid] = o
        lv = self.levels[o.side].get(o.price)
        if lv is None:
            lv = self.levels[o.side][o.price] = {}
        lv[oid] = True

    def _remove(self, oid: int):
        o = self.orders.pop(oid, None)
        if o is None:
            return None
        self.removed_prio[oid] = o.t_prio
        if len(self.removed_prio) > 200_000:      # keep only recently removed orders (O(1) eviction)
            self.removed_prio.popitem(last=False)
        lv = self.levels[o.side].get(o.price)
        if lv is not None:
            lv.pop(oid, None)
            if not lv:
                del self.levels[o.side][o.price]
        return o

    def reset_from_snapshot(self, data: dict, margin_us: int = 0):
        """Reset the book from a full L3 snapshot, then re-apply buffered events newer than it.

        Tardis inserts the REST snapshot into the stream ~1 s after the snapshot's own timestamp,
        so websocket events received in between must be re-applied after the reset.
        """
        self.levels = (SortedDict(), SortedDict())
        self.orders = {}
        t_snap = int(data["microtimestamp"])
        for side, key in ((0, "bids"), (1, "asks")):
            rows = sorted(((round(float(p) * 100), int(oid), float(a)) for p, a, oid in data[key]), key=lambda r: r[0])
            for px, oid, amt in rows:
                self._add(oid, Order(side, px, amt, order_time_us(oid), 0, True))
        self.valid = True
        self.last_event_id = None
        self.stats.snapshots += 1
        # for our periodic REST snapshots (same continuous connection) a 5 s margin before the snapshot
        # time is re-applied too, because REST snapshots can be slightly stale; re-applying is idempotent.
        # Never for snapshots taken after a reconnect: pre-gap creates whose deletes were lost would return.
        replay = [m for m in self.recent if int(m["data"]["microtimestamp"]) > t_snap - margin_us]
        self.replaying = True
        for m in replay:
            self.on_order(m["event"], m)
        self.replaying = False
        self.stats.counts["replayed_after_snapshot"] += len(replay)

    def _mid(self):
        b, a = self.best(0), self.best(1)
        if b is None or a is None:
            return None
        return 0.5 * (b + a)

    def _push_event(self, t, etype, side, px, amt, age, oid):
        if self.replaying:
            return
        mid = self._mid()
        if mid is None or abs(px - mid) / mid * 1e4 > NEAR_BPS:
            return
        i = self.ev_n % EV_CAP
        self.ev_t[i], self.ev_type[i], self.ev_side[i], self.ev_px[i] = t, etype, side, px
        self.ev_amt[i], self.ev_age[i], self.ev_oid[i] = amt, age, oid
        self.ev_dead[i] = False
        self.ev_n += 1
        if etype == CREATE:
            self.create_slot[oid] = self.ev_n - 1
            if len(self.create_slot) > 4 * EV_CAP:
                cutoff = self.ev_n - EV_CAP
                self.create_slot = {k: v for k, v in self.create_slot.items() if v >= cutoff}

    # ---------- message handlers ----------
    def on_order(self, ev: str, m: dict):
        d = m["data"]
        t = int(d["microtimestamp"])
        eid, pre = m.get("event_id"), m.get("pre_event_id")
        if not self.replaying and self.last_event_id is not None and pre is not None and pre != self.last_event_id:
            # one or more messages were lost: censor active virtual orders at the last time the feed was
            # known complete, pause sampling for 60 s, and keep the book (a lost message affects one order)
            self.stats.chain_breaks += 1
            self.censor_all(self.clock)
            self.quarantine_until = max(t, self.clock) + QUARANTINE_US
            self.last_break_t = max(t, self.clock)
            last = self.recent[-1] if self.recent else None
            self.recent.clear()                 # never re-apply pre-gap events after a later snapshot
            if last is not None:
                self.recent.append(last)
        if eid is not None:
            self.last_event_id = eid
        elif not self.replaying:
            self.stats.counts["no_event_id"] += 1
        self.advance(t)
        oid = int(d["id"])
        side = int(d["order_type"])
        px = round(float(d["price"]) * 100)
        amt = float(d["amount"])
        if not self.replaying:
            self.stats.counts[ev] += 1
        if ev == "order_created":
            self._add(oid, Order(side, px, amt, t))
            self._push_event(t, CREATE, side, px, amt, 0, oid)
        elif ev == "order_changed":
            o = self.orders.get(oid)
            if o is None:
                self.stats.counts["changed_unknown"] += 1
                self._add(oid, Order(side, px, amt, order_time_us(oid)))
                return
            if px != o.price or amt > o.amount + 1e-12:   # price amend or size increase: loses priority
                self._remove(oid)
                o.price, o.amount, o.n_changes, o.t_prio = px, amt, o.n_changes + 1, t
                self._add(oid, o)
                self._push_event(t, AMEND, side, px, amt, t - o.t_created, oid)
            else:                                          # partial fill or size reduction: keeps priority
                o.amount, o.n_changes = amt, o.n_changes + 1
                self._push_event(t, PARTIAL, side, px, amt, t - o.t_created, oid)
        elif ev == "order_deleted":
            o = self._remove(oid)
            if o is None:
                if not self.replaying:
                    self.stats.counts["deleted_unknown"] += 1
                return
            filled = amt <= 1e-12          # Bitstamp sends amount 0 when an order is fully filled
            if not self.replaying:
                self.stats.counts["delete_filled" if filled else "delete_cancel"] += 1
                self.fill_time[oid] = t      # time the order left the book (for the FIFO sweep test)
                if not filled:
                    self.cancels_hist.append((t, side, px))
                    bb = self.best(side)
                    if bb is None or (px >= bb if side == 0 else px <= bb):
                        self.cancel_rows.append((t, side, px, o.amount))
            slot = self.create_slot.pop(oid, None)
            if not filled and t - o.t_created < FLEETING_US and not self.replaying:
                # fleeting quote: drop both its create and cancel tokens, keep only a rate feature
                if slot is not None and slot >= self.ev_n - EV_CAP and self.ev_oid[slot % EV_CAP] == oid:
                    self.ev_dead[slot % EV_CAP] = True
                mid = self._mid()
                if mid is not None and abs(px - mid) / mid * 1e4 <= NEAR_BPS:
                    self.fleeting_hist.append(t)
                    self.stats.counts["fleeting"] += 1
                return
            self._push_event(t, FILL_DELETE if filled else CANCEL, side, px, amt if not filled else o.amount, t - o.t_created, oid)

    def on_trade(self, d: dict):
        t = int(d["microtimestamp"])
        self.advance(t)
        px = round(float(d["price"]) * 100)
        amt = float(d["amount"])
        taker_buy = int(d["type"]) == 0
        maker_side = 1 if taker_buy else 0
        maker = int(d["sell_order_id"] if taker_buy else d["buy_order_id"])
        self.stats.trades += 1
        self.trades_hist.append((t, maker_side, px, amt, maker))
        self.trade_rows.append((t, maker_side, px, amt))
        # FIFO test: was the maker the oldest order at its level?
        mo = self.orders.get(maker)
        if mo is None:
            if maker in self.removed_prio:          # its fill-delete arrived before the trade message
                self.stats.counts["maker_already_removed"] += 1
            else:
                self.stats.trades_unknown_maker += 1
                self.unknown_maker_lag_ms.append((t - order_time_us(maker)) / 1e3)
        elif self.valid:
            lv = self.levels[mo.side].get(mo.price)
            if lv:
                keys = list(lv.keys())
                pos = keys.index(maker) if maker in lv else -1
                self.stats.fifo_checked += 1
                self.stats.fifo_oldest += int(pos == 0)
                self.stats.fifo_in_first3 += int(0 <= pos < 3)
                if pos > 0:
                    self.fifo_pending.append((t, maker, keys[:min(pos, 10)]))
        self._push_event(t, TRADE, maker_side, px, amt, 0, maker)
        # virtual-order fills
        still = []
        if mo is not None:
            maker_prio = mo.t_prio
        else:
            maker_prio = self.removed_prio.get(maker, order_time_us(maker))
        for v in self.active[maker_side]:
            if t <= v.t0:                  # late-arriving trade from before this virtual order existed
                still.append(v)
                continue
            through = px < v.price if v.side == 0 else px > v.price
            # the incoming order reached V's position only if it hit an order queued after V joined
            if through or (px == v.price and maker_prio > v.t0):
                self.rows[v.row]["duration"] = (t - v.t0) / 1e6
                self.rows[v.row]["event"] = 1
            else:
                still.append(v)
        self.active[maker_side] = still

    # ---------- time, sampling, censoring ----------
    def mark_crossed(self, t: int):
        """First time an opposite order rests at or through V's price in an uncrossed book.

        A sell order resting at or below a virtual bid V could only rest there because V does not
        exist; had V been real, that order (unless post-only) would have executed against V.
        """
        b, a = self.best(0), self.best(1)
        if b is None or a is None or b >= a or (a > self.vlim[0] and b < self.vlim[1]):
            return
        for vside, opp in ((0, a), (1, b)):
            for v in self.active[vside]:
                row = self.rows[v.row]
                if t > v.t0 and (opp <= v.price if vside == 0 else opp >= v.price) and row["cross_duration"] != row["cross_duration"]:
                    row["cross_duration"] = (t - v.t0) / 1e6

    def censor_all(self, t: int):
        for side in (0, 1):
            for v in self.active[side]:
                self.rows[v.row]["duration"] = (t - v.t0) / 1e6
                self.rows[v.row]["event"] = 0
                self.rows[v.row]["gap_censored"] = 1
            self.active[side] = []

    def invalidate(self, t: int):
        """Book is unreliable: censor active virtual orders at t."""
        for side in (0, 1):
            for v in self.active[side]:
                self.rows[v.row]["duration"] = (t - v.t0) / 1e6
                self.rows[v.row]["event"] = 0
                self.rows[v.row]["gap_censored"] = 1
            self.active[side] = []
        self.valid = False
        # events from before the gap must never be re-applied after the next snapshot
        last = self.recent[-1] if self.recent else None
        self.recent.clear()
        if last is not None:
            self.recent.append(last)

    def advance(self, t: int):
        if t <= self.clock:
            return
        if self.next_sample is None:
            self.next_sample = (t // 1_000_000 + 1) * 1_000_000
        while self.next_sample <= t:
            self.sample(self.next_sample)
            self.next_sample += 1_000_000
        self.clock = t

    def censor_expired(self, now: int):
        for side in (0, 1):
            keep = []
            for v in self.active[side]:
                if now - v.t0 >= HORIZON_US:
                    self.rows[v.row]["duration"] = HORIZON_US / 1e6
                    self.rows[v.row]["event"] = 0
                else:
                    keep.append(v)
            self.active[side] = keep
            pxs = [v.price for v in keep]
            self.vlim[side] = (max(pxs) if side == 0 else min(pxs)) if pxs else (-1 if side == 0 else 1 << 62)

    def sample(self, s: int):
        self.censor_expired(s)
        self.stats.seconds_total += 1
        if not self.valid:
            self.stats.seconds_invalid += 1
            return
        if s < self.quarantine_until:
            self.stats.seconds_quarantine += 1
            return
        b, a = self.best(0), self.best(1)
        crossed_now = b is not None and a is not None and b >= a
        persisted, self.crossed_prev = crossed_now and self.crossed_prev, crossed_now
        while persisted and b is not None and a is not None and b >= a:
            # a cross that persists across samples: remove stale snapshot orders and orders that predate
            # the last lost message (ghosts whose delete was lost)
            stale = [o for side, p in ((0, b), (1, a)) for o in self.levels[side][p]
                     if self.orders[o].from_snapshot or self.orders[o].t_prio <= self.last_break_t]
            if not stale:
                break
            for o in stale:
                self._remove(o)
            self.stats.counts["stale_snapshot_removed"] += len(stale)
            b, a = self.best(0), self.best(1)
        if b is None or a is None or b >= a:
            self.stats.seconds_crossed += 1
            return
        mid = 0.5 * (a + b)
        self.mid_hist.append((s, mid))
        while self.mid_hist and s - self.mid_hist[0][0] > 60_000_000:
            self.mid_hist.popleft()
        while self.trades_hist and s - self.trades_hist[0][0] > 60_000_000:
            self.trades_hist.popleft()
        while self.cancels_hist and s - self.cancels_hist[0][0] > 10_000_000:
            self.cancels_hist.popleft()
        while self.fleeting_hist and s - self.fleeting_hist[0] > 10_000_000:
            self.fleeting_hist.popleft()
        snap_idx = len(self.snap_rows)
        self.snap_rows.append(self.l2_snapshot(mid))
        mids = np.array([x[1] for x in self.mid_hist])
        vol60 = float(np.std(np.diff(np.log(mids))) * 1e4) if len(mids) > 2 else 0.0
        for side, p in ((0, b), (1, a)):
            lv = self.levels[side][p]
            ahead = list(lv.keys())
            amts = np.array([self.orders[o].amount for o in ahead])
            ages = np.maximum(0.0, np.array([(s - self.orders[o].t_created) / 1e6 for o in ahead]))
            opp_best = a if side == 0 else b
            q_vol = float(amts.sum())
            opp_vol = float(sum(self.orders[o].amount for o in self.levels[1 - side][opp_best]))
            tv_side = sum(x[3] for x in self.trades_hist if x[1] == side)          # volume traded against this side, 60 s
            tv_at_p = sum(x[3] for x in self.trades_hist if x[1] == side and x[2] == p)
            n_tr = len(self.trades_hist)
            canc_p = sum(1 for x in self.cancels_hist if x[1] == side and x[2] == p)
            row = dict(day=self.day, t=s, side=side, price=p, mid=mid,
                       spread_bps=(a - b) / mid * 1e4, q_n=len(ahead), q_vol=q_vol, opp_vol=opp_vol,
                       imbalance=(q_vol - opp_vol) / (q_vol + opp_vol + 1e-12),
                       vol60_bps=vol60, tv_side_60=tv_side, tv_at_p_60=tv_at_p, n_trades_60=n_tr,
                       cancels_at_p_10=canc_p, fleeting_10=len(self.fleeting_hist),
                       q_age_mean=float(ages.mean()), q_age_max=float(ages.max()),
                       q_frac_old60=float(amts[ages > 60].sum() / (q_vol + 1e-12)),
                       q_snapshot_frac=float(np.mean([self.orders[o].from_snapshot for o in ahead])),
                       hour=(s // 3_600_000_000) % 24, snap_idx=snap_idx,
                       duration=np.nan, event=0, gap_censored=0, cross_duration=np.nan)
            self.rows.append(row)
            self.ev_tokens.append(self.event_tokens(s, side, p, frozenset(ahead)))
            self.q_tokens.append(self.queue_tokens(s, ahead, amts, ages))
            self.active[side].append(Virtual(side, p, s, frozenset(ahead), len(self.rows) - 1))
            self.vlim[side] = max(self.vlim[side], p) if side == 0 else min(self.vlim[side], p)

    def l2_snapshot(self, mid: float) -> np.ndarray:
        out = np.zeros((2, N_LEVELS, 2), np.float32)    # side, level, (offset bps, log volume)
        for side in (0, 1):
            lv = self.levels[side]
            keys = lv.keys()
            n = len(lv)
            idx = range(n - 1, max(n - 1 - N_LEVELS, -1), -1) if side == 0 else range(min(N_LEVELS, n))
            for j, k in enumerate(idx):
                price = keys[k]
                vol = sum(self.orders[o].amount for o in lv[price])
                out[side, j, 0] = (price - mid) / mid * 1e4
                out[side, j, 1] = math.log1p(vol * 100)
        return out.reshape(-1)

    def event_tokens(self, s: int, side: int, p: int, ahead: frozenset) -> np.ndarray:
        out = np.zeros((K_EVENTS, len(EVENT_FEATURES)), np.float32)
        m = min(self.ev_n, EV_CAP)
        if m == 0:
            return out.astype(np.float16)
        idx = (np.arange(self.ev_n - m, self.ev_n)) % EV_CAP
        idx = idx[~self.ev_dead[idx]][-K_EVENTS:]
        n = len(idx)
        if n == 0:
            return out.astype(np.float16)
        sgn = 1.0 if side == 0 else -1.0
        tt = self.ev_type[idx].astype(np.float32)
        out[-n:, 0] = tt + 1                                   # 0 = padding
        out[-n:, 1] = (self.ev_side[idx] == side)
        out[-n:, 2] = np.clip((self.ev_px[idx] - p) / p * 1e4 * sgn, -NEAR_BPS, NEAR_BPS)
        out[-n:, 3] = np.log1p(self.ev_amt[idx] * 100)
        out[-n:, 4] = np.log1p((s - self.ev_t[idx]) / 1e3)
        out[-n:, 5] = [o in ahead for o in self.ev_oid[idx]]
        out[-n:, 6] = np.log1p(self.ev_age[idx] / 1e3)
        out[-n:, 7] = (self.ev_type[idx] == TRADE) & (self.ev_side[idx] == side)
        return out.astype(np.float16)

    def queue_tokens(self, s, ahead, amts, ages) -> np.ndarray:
        out = np.zeros((Q_ORDERS, len(QUEUE_FEATURES) + 1), np.float32)   # last column = present flag
        m = min(len(ahead), Q_ORDERS)
        if m:
            cum = np.concatenate([[0.0], np.cumsum(amts)])[:m]
            out[:m, 0] = np.log1p(amts[:m] * 100)
            out[:m, 1] = np.log1p(ages[:m])
            out[:m, 2] = np.log1p(np.arange(m))
            out[:m, 3] = np.log1p(cum * 100)
            out[:m, 4] = [self.orders[o].n_changes > 0 for o in ahead[:m]]
            out[:m, 5] = 1
        return out.astype(np.float16)

    # ---------- driver ----------
    def run(self, day_dir: Path, messages=None):
        for m in (messages if messages is not None else iter_messages(day_dir)):
            ev = m.get("event")
            ch = m.get("channel", "")
            if ev == "snapshot" and ch.startswith("live_orders"):
                # a fresh snapshot resets the book; active virtual orders keep their (still valid) queues
                if m.get("reason") != "periodic":
                    # a reconnect snapshot follows a feed gap whose trades are lost: censor active orders
                    self.censor_all(self.clock)
                self.reset_from_snapshot(m["data"], SNAPSHOT_MARGIN_US if m.get("reason") == "periodic" else 0)
            elif ch.startswith("live_orders") and ev in ("order_created", "order_changed", "order_deleted"):
                self.recent.append(m)
                self.on_order(ev, m)
                self.mark_crossed(int(m["data"]["microtimestamp"]))
            elif ev == "trade":
                self.on_trade(m["data"])
        # censor whatever is still active at the end of the data
        for side in (0, 1):
            for v in self.active[side]:
                self.rows[v.row]["duration"] = (self.clock - v.t0) / 1e6
                self.rows[v.row]["event"] = 0
                self.rows[v.row]["gap_censored"] = 1   # end of file: not a full 300 s window
            self.active[side] = []


def iter_recorded(raw_root: Path, pair: str, day: str):
    """Our recorder's files for one UTC day: merge the websocket stream with the REST snapshots
    by local receive time and present snapshots in the same form as Tardis' generated ones."""
    import heapq

    def stream(sub, is_snap):
        for f in sorted((raw_root / sub / day).glob("*.jsonl.gz")):
            with gzip.open(f, "rb") as fh:
                try:
                    for line in fh:
                        if len(line) < 3 or not line.endswith(b"\n"):
                            continue
                        rec = orjson.loads(line)
                        yield rec["t"], is_snap, rec["m"]
                except EOFError:          # file still being written (or recorder stopped mid-hour)
                    pass

    for _t, is_snap, raw in heapq.merge(stream(f"bitstamp_{pair}", False), stream(f"bitstamp_snapshot_{pair}", True)):
        m = orjson.loads(raw)
        if is_snap:
            yield {"event": "snapshot", "channel": f"live_orders_{pair}", "data": m["book"], "reason": m.get("reason")}
        else:
            yield m


def process_day(day_dir: str, out_dir: str, recorded_pair: str | None = None) -> dict:
    """Replay one day. Tardis: day_dir = data/tardis/bitstamp_<pair>/<YYYY-MM-DD>.
    Our recorder: day_dir = data/raw/<YYYYMMDD> (a virtual path) with recorded_pair set."""
    day_dir, out_dir = Path(day_dir), Path(out_dir)
    day = day_dir.name
    out_dir.mkdir(parents=True, exist_ok=True)
    r = Replayer(day)
    if recorded_pair:
        r.run(day_dir, messages=iter_recorded(day_dir.parent, recorded_pair, day))
    else:
        r.run(day_dir)
    df = pd.DataFrame(r.rows)
    df.to_parquet(out_dir / f"{day}_labels.parquet")
    np.save(out_dir / f"{day}_events.npy", np.stack(r.ev_tokens) if r.ev_tokens else np.zeros((0, K_EVENTS, len(EVENT_FEATURES)), np.float16))
    np.save(out_dir / f"{day}_queue.npy", np.stack(r.q_tokens) if r.q_tokens else np.zeros((0, Q_ORDERS, len(QUEUE_FEATURES) + 1), np.float16))
    pd.DataFrame(r.trade_rows, columns=["t", "maker_side", "price", "amount"]).to_parquet(out_dir / f"{day}_trades.parquet")
    pd.DataFrame(r.cancel_rows, columns=["t", "side", "price", "amount"]).to_parquet(out_dir / f"{day}_cancels.parquet")
    np.save(out_dir / f"{day}_snaps.npy", np.stack(r.snap_rows) if r.snap_rows else np.zeros((0, 4 * N_LEVELS), np.float32))
    st = r.stats
    # FIFO with sweeps: a non-oldest maker is consistent if every order ahead of it was filled within 5 ms
    sweep_ok = sum(1 for t, mk, ahead in r.fifo_pending
                   if all(r.fill_time.get(o, 10**19) <= t + 5_000 for o in ahead))
    lag = np.array(r.unknown_maker_lag_ms) if r.unknown_maker_lag_ms else np.zeros(1)
    stats_extra = dict(fifo_sweep_consistent=st.fifo_oldest + sweep_ok,
                       unknown_maker_lag_ms_q50=float(np.median(lag)), unknown_maker_lag_ms_q90=float(np.quantile(lag, 0.9)),
                       fleeting_dropped=st.counts.get("fleeting", 0),
                       chain_monitored=st.counts.get("no_event_id", 0) == 0,
                       maker_already_removed=st.counts.get("maker_already_removed", 0))
    stats = dict(**stats_extra, day=day, **{f"n_{k}": v for k, v in st.counts.items()}, trades=st.trades,
                 trades_unknown_maker=st.trades_unknown_maker, fifo_checked=st.fifo_checked,
                 fifo_oldest=st.fifo_oldest, fifo_in_first3=st.fifo_in_first3, chain_breaks=st.chain_breaks,
                 snapshots=st.snapshots, seconds_total=st.seconds_total, seconds_invalid=st.seconds_invalid,
                 seconds_crossed=st.seconds_crossed, seconds_quarantine=st.seconds_quarantine, examples=len(df),
                 near_events=r.ev_n)
    (out_dir / f"{day}_stats.json").write_text(json.dumps(stats, indent=1))
    return stats


if __name__ == "__main__":
    import sys
    print(process_day(sys.argv[1], sys.argv[2]))
