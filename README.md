# Order-Flow Fill Probability

APS360 (University of Toronto, Fall 2026) individual project by Adam Abdalla.

The model predicts the fill-time distribution of a limit order posted at the best bid or ask, P(fill ≤ t) for t in 0.1–300 s, from Bitstamp's public order-by-order (L3) feed.

- The **Order-Flow Transformer** reads two kinds of input: the last 256 near-touch order events, and the orders already queued ahead (their sizes and ages). Aggregated order-book snapshots hide that queue composition.
- It outputs 20 discrete hazards and is trained with the censored discrete-time negative log-likelihood.
- Baselines:
  - **B1** Kaplan–Meier
  - **B2** Cox model, on L2 features and on L2 + L3 summaries
  - **B3** queue-depletion rule
  - **B4** snapshot conv-Transformer

## Data

| Source | Use |
|---|---|
| [Tardis.dev](https://docs.tardis.dev/historical-data-details/bitstamp) free first-of-month archive of Bitstamp `live_orders` + `live_trades` (BTC/USD, 2025-12-01 … 2026-11-01) | train / validation / test |
| Our own recorder (`collect/record.py`), Bitstamp + Bitfinex, from 2026-11-18 | never-seen test data |

The repository holds code, statistics and results only. Raw and processed market data are not redistributed.

## Reproduce

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python websockets httpx orjson numpy pandas pyarrow sortedcontainers torch lifelines scikit-survival matplotlib pytest

.venv/bin/python collect/tardis_download.py 2025-12-01 2026-11-01 --pair btcusd   # 12 days, ~6 GB
.venv/bin/python scripts/build_dataset.py --jobs 5          # replay, quality checks, labels, tokens -> data/processed
.venv/bin/python scripts/data_stats.py                      # results/data_stats.json
.venv/bin/python scripts/baselines.py                       # B1-B3 -> results/baselines.json
.venv/bin/python scripts/cox_val.py                        # Cox validation NLL (learning-curve reference)
.venv/bin/python scripts/train.py --model snap --tag b4_matched  --epochs 4 --dropout 0.15 --wd 0.05
.venv/bin/python scripts/train.py --model oft  --tag oft_nopre   --epochs 4 --dropout 0.15 --wd 0.05
.venv/bin/python scripts/train.py --model oft  --tag oft_pre --pretrain-epochs 1 --epochs 4 --dropout 0.15 --wd 0.05
.venv/bin/python scripts/train.py --model oft  --tag oft_noqueue  --epochs 4 --dropout 0.15 --wd 0.05 --no-queue
.venv/bin/python scripts/train.py --model oft  --tag oft_noevents --epochs 4 --dropout 0.15 --wd 0.05 --no-events
.venv/bin/python scripts/group_stats.py                     # young/old and short/long queue fill rates
.venv/bin/python scripts/eval_new_data.py --refresh 20261118 20261119   # never-seen data from collect/record.py
.venv/bin/python scripts/make_figures.py data results       # report/figs/*.pdf
.venv/bin/python scripts/make_tables.py report              # LaTeX tables + new-data macros
.venv/bin/python -m pytest tests -q                         # metric tests
```

## Results so far (test days Oct 1 + Nov 1, 113k virtual orders)

| Model | C^td | NLL | IBS |
|---|---|---|---|
| B1 Kaplan–Meier | 0.506 | 2.768 | 0.195 |
| B2 Cox, L2 features | 0.649 | 2.677 | 0.176 |
| B2+L3 Cox | 0.668 | 2.658 | 0.172 |
| B4 snapshot conv-Transformer | 0.659 | 2.615 | 0.174 |
| **OFT** (900,244 params) | **0.677** | **2.576** | **0.170** |
| OFT, no event tokens | 0.669 | 2.604 | 0.173 |

- Removing the queue tokens changes nothing (the context token already holds queue summaries). Removing the event tokens drops C^td to the Cox level.
- One epoch of masked-event pretraining lowers validation NLL but not test NLL.
- On never-seen recorded data (`results/new_data.json`) the OFT also leads.

**In progress:** final test on full recorded days, longer training and pretraining, trade-only labels, Bitfinex as a second venue.

## Layout

```
collect/    live recorder (Bitstamp L3 + Bitfinex raw books) and Tardis downloader
fillprob/   replay.py (book rebuild, gap/FIFO checks, virtual-order labels, tokens),
            data.py (splits, features), models.py, bins.py + metrics.py (C-index, IBS, calibration)
scripts/    dataset build, statistics, baselines, training, new-data evaluation, figures, tables
tests/      metric tests
results/    statistics and test metrics (JSON)
```
