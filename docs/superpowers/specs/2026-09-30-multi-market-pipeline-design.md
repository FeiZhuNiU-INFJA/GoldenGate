# Multi-market quant pipeline rewrite (2026-09-30)

## Goal

Replace the legacy Tushare + zigzag-label + small Transformer stack with a clean Akshare-based pipeline: CN A-shares (ex-BJ) + Hang Seng + S&P 500, dual-horizon excess-return classification, shared encoder with per-market heads.

## Decisions

- Labels: 5d / 20d forward excess return vs CSI300 / HSI / SPX; thresholds ±1.5% / ±4%
- Model: Pre-Norm Transformer encoder (swappable) + market embedding + per-market `5d`/`20d` heads
- Success: PR/F1 vs always-neutral and momentum baselines + rough weekly Top-N L/S backtest
- Storage: parquet under `dataset/{cn,hk,us}/`
- CLI entrypoints under `scripts/`; signal script named `emit_signals.py` to avoid shadowing stdlib `signal`

## Non-goals

- LLM / news features
- Production execution, costs, limit-up handling
- Migrating old checkpoints
