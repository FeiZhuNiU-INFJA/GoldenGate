# CLAUDE.md

Guidance for working in this repository.

## Project Overview

Quantitative trading research system for **CN A-shares (主板/科创/创业), Hang Seng, S&P 500, and Nasdaq-100**. Downloads daily bars via **Akshare**, labels with **5d/20d forward excess returns** vs market benchmarks, trains a **shared Transformer encoder** with **per-market dual classification heads**, and evaluates with classification metrics plus a rough Top-N long/short backtest.

## Common Commands

```bash
conda activate extreme_quant   # or: conda env create -f environment.yml
pip install -r requirements.txt

python scripts/download_data.py --markets cn hk us
python scripts/build_labels.py --markets cn hk us --summary
python scripts/train.py
python scripts/evaluate.py --split val
python scripts/emit_signals.py --date YYYY-MM-DD

pytest tests/
```

Smoke (few symbols):

```bash
python scripts/make_synthetic_data.py   # offline, no Akshare
python scripts/download_data.py --max-symbols 5
python scripts/build_labels.py --max-symbols 5
python scripts/train.py --epochs 2 --max-symbols 5 --batch-size 64
```

If Eastmoney requests fail with `ProxyError`, unset `http_proxy`/`https_proxy` or keep `EXTREME_QUANT_KEEP_PROXY` unset (Akshare client clears proxies by default).

## Architecture

1. **Download** (`data/download.py`, `data/market_client.py`) → Baostock (CN) + yfinance/Sina (HK/US/Nasdaq-100) → `dataset/{market}/bars/*.parquet`, `universe.csv`, `benchmark.parquet`, `calendar.parquet`. Nasdaq-100 names that are also in the S&P 500 share `dataset/us/bars/`.
2. **Labels** (`labels/excess_return.py`) → `dataset/{market}/labeled/*.parquet` with `y_5d`, `y_20d`
3. **Train** (`train/dataset.py`, `train/trainer.py`, `models/multi_head.py`) → `checkpoints/best_val_loss.pt`
4. **Eval / signal** (`scripts/evaluate.py`, `scripts/emit_signals.py`)

### Label mapping

- `0` neutral, `1` buy, `2` sell
- Thresholds in `config/settings.py` (`LABEL_THRESHOLDS`)
- Benchmarks: CN `000300`, HK `HSI`, US S&P 500

### Features

`open, high, low, close, volume, amount` — window-normalized by first-day open / volume / amount (`SEQ_LEN=128`).

### Model

- Shared Pre-Norm Transformer encoder (`models/transformer_encoder.py`)
- Market embedding + per-market heads for `5d` and `20d`
- Inference score: `0.4*(P_buy-P_sell)_5d + 0.6*(P_buy-P_sell)_20d`

### Time splits

- Train: 2015-01-01 → 2022-12-31
- Val: 2023-01-01 → 2023-12-31
- Test: 2024-01-01 → latest

Partition by **label date** `t`; inputs are only `[t-seq_len+1, t]`.

## Conventions

- Prefer parquet under `dataset/`; never commit raw market dumps
- Keep encoder swappable via `EncoderBase`
- Akshare calls must go through `data/akshare_client.py` (retry + sleep)
- Do not reintroduce Tushare tokens
