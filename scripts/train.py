#!/usr/bin/env python3
"""Train multi-market dual-horizon classifier."""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from train.trainer import TrainConfig, train

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--max-symbols", type=int, default=None, help="per market, for smoke runs")
    p.add_argument("--device", default=None, help="cuda | cuda:0 | mps | cpu (default: auto)")
    p.add_argument("--num-workers", type=int, default=None, help="DataLoader workers (CUDA default 4)")
    args = p.parse_args()

    cfg = TrainConfig()
    if args.epochs is not None:
        cfg.epochs = args.epochs
    if args.batch_size is not None:
        cfg.batch_size = args.batch_size
    if args.lr is not None:
        cfg.lr = args.lr
    if args.max_symbols is not None:
        cfg.max_symbols_per_market = args.max_symbols
    if args.device is not None:
        cfg.device = args.device
    if args.num_workers is not None:
        cfg.num_workers = args.num_workers

    path = train(cfg)
    print(f"best checkpoint: {path}")


if __name__ == "__main__":
    main()
