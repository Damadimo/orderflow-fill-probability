# Order-Flow Fill Probability

APS360 (University of Toronto, Fall 2026) individual project by Adam Abdalla.

The project predicts the probability that a limit order posted at the best bid or ask fills within 1–60 seconds. A Transformer reads the order-by-order (L3) event stream, so it sees queue composition that aggregated order-book snapshots hide. It is pretrained with self-supervision on unlabeled order flow, then fine-tuned as a discrete-time survival model.

**Status:** proposal stage (October 2026).

## Data

These public websocket feeds are recorded with collectors in this repo. No API key is required.

| Venue | Channels | Content |
|---|---|---|
| Bitstamp | `live_orders_{btcusd,ethusd}`, `live_trades_*` | Order create/change/delete events with order IDs; trades carry buy/sell order IDs |
| Bitfinex | `book` (prec `R0`), `trades` | Raw order-level book with CRC32 checksums (cross-venue test set) |

## Baselines

- Kaplan–Meier
- Cox proportional hazards
- Queue-depletion heuristic
- Order-book-snapshot convolutional Transformer

## Planned layout

```
collect/     websocket recorders (raw JSON lines + receive timestamps)
book/        order-book reconstruction, checksum and sequence-gap checks
labels/      virtual-order fill labelling (right-censored)
models/      event-token Transformer, pretraining, survival head, baselines
eval/        C-index, integrated Brier score, calibration, replay test
```
