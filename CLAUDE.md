# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Quantitative trading research for **CN A-shares (主板/科创/创业), Hang Seng, S&P 500, and Nasdaq-100** (markets: `cn`, `hk`, `us`, `ndx`). Two parallel model pipelines over daily bars:

- **Classification**: shared Transformer encoder with per-market dual heads; 5d/20d forward excess return vs benchmark → buy/neutral/sell.
- **Ranking**: per-market, per-horizon (5/10/20d) LightGBM LambdaRank on cross-sectional labels. S&P and Nasdaq daily trackers record each horizon's top-5 intersection of three seed models.

Nasdaq-100 names that are also in the S&P 500 share `dataset/us/bars/`; the rest live under `dataset/ndx/bars/`. `dataset/` is never committed.

## Environment

Python 3.12, conda env `extreme_quant`. `environment.yml` uses Tsinghua mirrors and does **not** include LightGBM — ranking also needs `requirements.txt`:

```bash
conda env create -f environment.yml   # existing env: conda env update -f environment.yml --prune
conda activate extreme_quant
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

Device: MPS on Apple Silicon, CUDA on NVIDIA, else CPU (`--device` overrides).

## Common Commands

```bash
# Data (existing parquet skipped by default; --force re-downloads when 前复权 changed outside the incremental window)
python scripts/download_data.py --markets cn hk us ndx --workers 8
python scripts/update_data.py --markets cn hk us ndx --end YYYYMMDD   # incremental
python scripts/build_labels.py --markets cn hk us ndx --summary

# Classification
python scripts/train.py --epochs 20
python scripts/evaluate.py --checkpoint checkpoints/best_val_loss.pt --split val
python scripts/emit_signals.py --date YYYY-MM-DD

# Ranking
python scripts/compare_rank_horizons.py --markets cn hk us ndx   # --rebuild-panel to refresh cache
python scripts/compare_rank_ensemble.py --markets us ndx         # 3-seed ensemble → checkpoints/ensemble/
python scripts/update_us_recommendations.py                      # refresh docs/live/ trackers

# Tests
pytest tests/
pytest tests/test_rank_labels.py          # single file

# Offline smoke (no network)
python scripts/make_synthetic_data.py
python scripts/build_labels.py --max-symbols 5
python scripts/train.py --epochs 2 --max-symbols 5 --batch-size 64
```

### Daily US intersection (S&P + Nasdaq only, never retrain)

`--end` must be the last **completed** US session (after 16:00 America/New_York; weekends stay on Friday). Run one update at a time; don't raise `--workers` above 8.

```bash
EQ_SKIP_YFINANCE=1 python scripts/update_data.py --markets us ndx --end YYYYMMDD
python scripts/update_us_recommendations.py
```

## Architecture

1. **Download** (`data/download.py`, `data/market_client.py`) → CN via Baostock (fallback Sina), HK/US/NDX via yfinance (fallback Sina/Eastmoney) → `dataset/{market}/bars/*.parquet`, `universe.csv`, `benchmark.parquet`, `calendar.parquet`. Feature-engineering caches: `dataset/panels/rank{h}.parquet`, `rank_multi.parquet`.
2. **Labels** (`labels/excess_return.py`, `labels/cross_section.py`) → `dataset/{market}/labeled/*.parquet`.
3. **Train** (`train/trainer.py`, `train/dataset.py`, `models/multi_head.py`) → `checkpoints/best_val_loss.pt`; rankers → `checkpoints/ensemble/ranker_{market}_h{5,10,20}_s{0,1,2}.txt`.
4. **Eval / signal** (`scripts/evaluate.py`, `scripts/emit_signals.py`, `eval/`, `scripts/update_us_recommendations.py`).

### Classification model

- Features: `open, high, low, close, volume, amount`, window-normalized by first-day open / volume / amount (`SEQ_LEN=128`).
- Shared Pre-Norm Transformer (`models/transformer_encoder.py`) + market embedding + per-market 5d/20d heads.
- Inference score: `0.4*(P_buy-P_sell)_5d + 0.6*(P_buy-P_sell)_20d`.
- Label mapping `0` neutral / `1` buy / `2` sell; thresholds in `config/settings.py` (`LABEL_THRESHOLDS`): 5d ±1.5%, 20d ±4%.
- Benchmarks: CN `000300`, HK `HSI`, US `.INX`, NDX `.NDX`.

### Ranking pipeline

- Features (`train/rank_features.py`): 7 causal tabular features — `ret_5/20/60`, `vol_20`, `activity_ratio_20`, `range_pct`, `close_loc`. Nothing reads future prices.
- Cross-sectional labels (`labels/cross_section.py`): per market-day z-score of forward excess return; relevance grades by rank — 1–5 → 2, 6–15 → 1, rest 0 (gains 2^r−1); groups under 20 names dropped.
- Protocol (`train/rank_protocol.py`): fit on `t < 2024-01-01`; tune tree count + score sign on 2024-01-01 → 2024-05; refit on `t < 2024-06-01`; `t ≥ 2024-06-01` is report-only — never used to choose hyperparameters.
- Daily recommendation = intersection of each of 3 seeds' top-5, per horizon; empty intersection = 空仓.

### Recommendation trackers

`scripts/update_us_recommendations.py` rewrites `docs/live/us-intersection.md`, `docs/live/ndx-intersection.md`, the ledgers `docs/live/*-intersection.json`, and `docs/live/intersection-data.js` (read by the fixed `docs/live/intersection.html` — the page itself is never rewritten). A date with a thin cross-section (us < 450 names, ndx < 80) is left unrecorded; dates before each book's start (us 2026-10-01, ndx 2026-10-02) are never backfilled.

### Time splits (by label date `t`; inputs only `[t-seq_len+1, t]`)

- Classification: train 2015-01-01 → 2022-12-31, val 2023, test 2024-01-01 → latest.
- Ranking: see protocol above.

## Conventions

- Prefer parquet under `dataset/`; never commit raw market dumps or `dataset/`.
- Market-data fetches go through `data/market_client.py` (retry + sleep via `data/akshare_client.py`, which also clears proxy env vars). Do not reintroduce Tushare.
- Proxy: an Eastmoney `ProxyError` means keep `EXTREME_QUANT_KEEP_PROXY` unset (proxies are cleared by default; set it to `1` only when the user explicitly wants the proxy kept). `EQ_SKIP_YFINANCE=1` skips yfinance and forces the fallback path.
- `docs/live/*.md` are regenerated on every run — hand edits are overwritten; don't edit ledgers/JSON to force a thin day in.
- Keep the encoder swappable via `EncoderBase` (`models/encoder_base.py`).
- Experiment write-ups: `docs/experiments/YYYY-MM-DD-*.md`; eval JSON dumps: `reports/`. Operational Cursor skills: `.cursor/skills/` (setup-environment, daily-us-intersection).
