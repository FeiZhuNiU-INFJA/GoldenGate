# Extreme Quant (v2)

Multi-market daily bar classifier: **CN A-shares (主板/科创/创业) + Hang Seng + S&P 500**.

Pipeline: Baostock (A股) + yfinance (港/美) download → 5d/20d excess-return labels → shared Transformer encoder with per-market dual heads → PR/F1 + rough Top-N backtest.

## Setup

```bash
# Recommended: conda env (environment.yml 已用清华源)
conda env create -f environment.yml
conda activate extreme_quant

# Or pip-only in an existing env (清华 PyPI)
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

Python **3.12** is the supported version (see `environment.yml`).

On **Apple Silicon (M1/M2/M3/M4)**, PyTorch uses **MPS (Metal)** automatically. Check:

```bash
python -c "import torch; print(torch.__version__, 'mps=', torch.backends.mps.is_available())"
```

## Commands

```bash
# Offline smoke (no network)
python scripts/make_synthetic_data.py
python scripts/build_labels.py --markets cn hk us --force --summary
python scripts/train.py --epochs 2 --max-symbols 5 --batch-size 64
python scripts/evaluate.py --split val --max-symbols 5
python scripts/emit_signals.py --date 2023-06-30 --markets cn --top-n 5

# 1) Download (Baostock CN + yfinance HK/US). Use --max-symbols for a smoke run.
# Resume-friendly: omit --force to skip already-saved parquet bars.
# If Wikipedia blocked for HK/US universe, client falls back to a short mega-cap list.
python scripts/download_data.py --markets cn hk us --max-symbols 5

# Full download (CN uses process pool; resume-friendly — omit --force)
python scripts/download_data.py --markets cn hk us --workers 8

# 2) Build labels
python scripts/build_labels.py --markets cn hk us --summary

# 3) Train
python scripts/train.py --epochs 20

# Smoke train
python scripts/train.py --epochs 2 --max-symbols 5 --batch-size 64

# 4) Evaluate (metrics + baselines + weekly Top-N backtest)
python scripts/evaluate.py --checkpoint checkpoints/best_val_loss.pt --split val

# 5) Signals for a date
python scripts/emit_signals.py --date 2024-06-28 --markets cn hk us
```

## Layout

- `config/` — paths, thresholds, date splits, model dims
- `data/` — Akshare client, download, schema, store, calendar
- `labels/` — forward excess-return → buy/neutral/sell
- `models/` — swappable encoder + multi-market heads
- `train/` — dataset + trainer
- `eval/` — metrics, baselines, backtest
- `scripts/` — CLI entrypoints

## Labels

Relative to market benchmark (CSI 300 / HSI / S&P 500):

| Horizon | Buy | Sell |
|---------|-----|------|
| 5d | excess ≥ +1.5% | excess ≤ -1.5% |
| 20d | excess ≥ +4% | excess ≤ -4% |

Classes: `0=neutral`, `1=buy`, `2=sell`.

## Notes

- Data lands under `dataset/{cn,hk,us}/` as parquet (gitignored).
- Old Tushare / zigzag-label codebase was removed in this rewrite.
- Encoder is intentionally swappable via `models/encoder_base.py`.
