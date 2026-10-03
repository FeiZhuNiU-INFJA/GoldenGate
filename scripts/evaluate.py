#!/usr/bin/env python3
"""Evaluate checkpoint: classification metrics + baselines + rough Top-N backtest."""
from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config.device import get_device
from config.settings import DIR_CHECKPOINTS, DIR_REPORTS, ID_TO_MARKET, MARKETS, SCORE_WEIGHT_5D, SCORE_WEIGHT_20D
from eval.baselines import always_neutral, momentum_from_features
from eval.backtest import run_topn_backtest, summarize_backtest
from eval.metrics import summarize_predictions
from train.dataset import MultiMarketDataset
from train.trainer import load_model

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


@torch.no_grad()
def collect_predictions(model, dataset, device, batch_size: int = 512):
    y5_true, y5_pred = [], []
    y20_true, y20_pred = [], []
    xs = []
    metas = []  # market_id, symbol, date
    scores = []
    batch_x, batch_y5, batch_y20, batch_mid, batch_meta = [], [], [], [], []

    def flush():
        nonlocal batch_x, batch_y5, batch_y20, batch_mid, batch_meta
        if not batch_x:
            return
        x = torch.stack(batch_x).to(device)
        mid = torch.stack(batch_mid).to(device)
        y5 = torch.stack(batch_y5)
        y20 = torch.stack(batch_y20)
        probs = model.predict_proba(x, mid)
        pred5 = probs["5d"].argmax(dim=-1).cpu().numpy()
        pred20 = probs["20d"].argmax(dim=-1).cpu().numpy()
        score = model.score(x, mid, w5=SCORE_WEIGHT_5D, w20=SCORE_WEIGHT_20D).cpu().numpy()
        y5_true.append(y5.numpy())
        y20_true.append(y20.numpy())
        y5_pred.append(pred5)
        y20_pred.append(pred20)
        xs.append(x.cpu().numpy())
        scores.append(score)
        metas.extend(batch_meta)
        batch_x, batch_y5, batch_y20, batch_mid, batch_meta = [], [], [], [], []

    for i in tqdm(range(len(dataset)), desc="eval"):
        x, y5, y20, mid, symbol, date = dataset[i]
        batch_x.append(x)
        batch_y5.append(y5)
        batch_y20.append(y20)
        batch_mid.append(mid)
        batch_meta.append((int(mid.item()), symbol, date))
        if len(batch_x) >= batch_size:
            flush()
    flush()
    return {
        "y5_true": np.concatenate(y5_true),
        "y5_pred": np.concatenate(y5_pred),
        "y20_true": np.concatenate(y20_true),
        "y20_pred": np.concatenate(y20_pred),
        "x": np.concatenate(xs),
        "scores": np.concatenate(scores),
        "metas": metas,
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, default=str(DIR_CHECKPOINTS / "best_val_loss.pt"))
    p.add_argument("--split", default="val", choices=["train", "val", "test"])
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--max-symbols", type=int, default=None)
    p.add_argument("--top-n", type=int, default=20)
    p.add_argument("--device", default=None)
    args = p.parse_args()

    device = args.device or get_device()
    model = load_model(args.checkpoint, device=device)
    ds = MultiMarketDataset(args.split, max_symbols_per_market=args.max_symbols, with_meta=True)
    if len(ds) == 0:
        raise SystemExit(f"empty {args.split} dataset")

    out = collect_predictions(model, ds, device, batch_size=args.batch_size)
    report = {
        "split": args.split,
        "n": int(len(out["y5_true"])),
        "model_5d": summarize_predictions(out["y5_true"], out["y5_pred"]),
        "model_20d": summarize_predictions(out["y20_true"], out["y20_pred"]),
    }
    # baselines on 20d
    n = len(out["y20_true"])
    report["baseline_always_neutral_20d"] = summarize_predictions(out["y20_true"], always_neutral(n))
    mom = momentum_from_features(out["x"], threshold=0.04)
    report["baseline_momentum_20d"] = summarize_predictions(out["y20_true"], mom)

    print("=== model 5d ===")
    print(report["model_5d"]["report"])
    print("=== model 20d ===")
    print(report["model_20d"]["report"])
    print("=== always-neutral 20d macro_f1 ===", report["baseline_always_neutral_20d"]["macro_f1"])
    print("=== momentum 20d macro_f1 ===", report["baseline_momentum_20d"]["macro_f1"])
    print("=== model 20d macro_f1 ===", report["model_20d"]["macro_f1"])

    # Build weekly score maps per market for rough backtest on this split's dates
    by_market_date: dict[str, dict[pd.Timestamp, dict[str, float]]] = {m: defaultdict(dict) for m in MARKETS}
    for (mid, symbol, date), score in zip(out["metas"], out["scores"]):
        market = ID_TO_MARKET[mid]
        dt = pd.Timestamp(date)
        # keep Friday-ish: use all dates; rebalance filter weekly
        by_market_date[market][dt][symbol] = float(score)

    backtests = {}
    for market, date_map in by_market_date.items():
        if not date_map:
            continue
        # downsample to one date per week (Friday preference)
        dates = sorted(date_map.keys())
        weekly = {}
        for dt in dates:
            if dt.weekday() == 4 or dt == dates[-1]:
                weekly[dt] = date_map[dt]
        # if too few Fridays, take every 5th date
        if len(weekly) < 3:
            weekly = {dates[i]: date_map[dates[i]] for i in range(0, len(dates), 5)}
        bt = run_topn_backtest(market, weekly, top_n=args.top_n, hold_days=5)
        summary = summarize_backtest(bt)
        backtests[market] = summary
        print(f"=== backtest {market} ===", summary)

    report["backtest"] = backtests
    # Strip bulky text fields for json
    slim = {
        "split": report["split"],
        "n": report["n"],
        "model_5d_macro_f1": report["model_5d"]["macro_f1"],
        "model_20d_macro_f1": report["model_20d"]["macro_f1"],
        "baseline_neutral_20d_macro_f1": report["baseline_always_neutral_20d"]["macro_f1"],
        "baseline_momentum_20d_macro_f1": report["baseline_momentum_20d"]["macro_f1"],
        "backtest": backtests,
        "model_5d_report": report["model_5d"]["report"],
        "model_20d_report": report["model_20d"]["report"],
    }
    DIR_REPORTS.mkdir(parents=True, exist_ok=True)
    out_path = DIR_REPORTS / f"eval_{args.split}.json"
    out_path.write_text(json.dumps(slim, indent=2), encoding="utf-8")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
