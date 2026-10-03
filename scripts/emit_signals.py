#!/usr/bin/env python3
"""Emit long/short ranking for a given trade date."""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config.device import get_device
from config.settings import (
    DIR_CHECKPOINTS,
    FEATURE_COLS,
    MARKET_TO_ID,
    MARKETS,
    SCORE_WEIGHT_5D,
    SCORE_WEIGHT_20D,
    SEQ_LEN,
)
from data import store
from train.dataset import preprocess_window
from train.trainer import load_model

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def window_for_date(df: pd.DataFrame, trade_date: pd.Timestamp, seq_len: int) -> np.ndarray | None:
    dates = pd.to_datetime(df["trade_date"])
    idxs = np.where(dates == pd.Timestamp(trade_date))[0]
    if len(idxs) == 0:
        return None
    i = int(idxs[0])
    if i < seq_len - 1:
        return None
    feats = df[FEATURE_COLS].to_numpy(dtype=np.float64)[i - seq_len + 1 : i + 1]
    return preprocess_window(feats)


@torch.no_grad()
def score_market(model, market: str, trade_date: str, device: str, top_n: int) -> pd.DataFrame:
    dt = pd.Timestamp(trade_date)
    rows = []
    symbols = store.list_bar_symbols(market)
    # Prefer labeled files if present (same OHLCV)
    labeled = set(store.list_labeled_symbols(market))
    mid = MARKET_TO_ID[market]
    batch_x, batch_sym = [], []
    for sym in tqdm(symbols, desc=f"signal-{market}"):
        path = store.labeled_path(market, sym) if sym in labeled else store.resolve_bar_path(market, sym)
        if path is None:
            continue
        df = store.read_parquet(path)
        if df is None or df.empty:
            continue
        if any(c not in df.columns for c in FEATURE_COLS):
            continue
        w = window_for_date(df, dt, SEQ_LEN)
        if w is None:
            continue
        batch_x.append(w)
        batch_sym.append(sym)
        if len(batch_x) >= 256:
            x = torch.from_numpy(np.stack(batch_x)).to(device)
            mids = torch.full((x.size(0),), mid, dtype=torch.long, device=device)
            scores = model.score(x, mids, w5=SCORE_WEIGHT_5D, w20=SCORE_WEIGHT_20D).cpu().numpy()
            for s, sc in zip(batch_sym, scores):
                rows.append({"symbol": s, "market": market, "score": float(sc)})
            batch_x, batch_sym = [], []
    if batch_x:
        x = torch.from_numpy(np.stack(batch_x)).to(device)
        mids = torch.full((x.size(0),), mid, dtype=torch.long, device=device)
        scores = model.score(x, mids, w5=SCORE_WEIGHT_5D, w20=SCORE_WEIGHT_20D).cpu().numpy()
        for s, sc in zip(batch_sym, scores):
            rows.append({"symbol": s, "market": market, "score": float(sc)})
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out = out.sort_values("score", ascending=False).reset_index(drop=True)
    out["rank"] = np.arange(1, len(out) + 1)
    print(f"\n=== {market} TOP {top_n} ===")
    print(out.head(top_n).to_string(index=False))
    print(f"\n=== {market} BOTTOM {top_n} ===")
    print(out.tail(top_n).to_string(index=False))
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--date", required=True, help="YYYY-MM-DD")
    p.add_argument("--markets", nargs="+", default=list(MARKETS), choices=list(MARKETS))
    p.add_argument("--checkpoint", default=str(DIR_CHECKPOINTS / "best_val_loss.pt"))
    p.add_argument("--top-n", type=int, default=20)
    p.add_argument("--device", default=None)
    p.add_argument("--out", default=None, help="optional csv path")
    args = p.parse_args()

    device = args.device or get_device()
    model = load_model(args.checkpoint, device=device)
    frames = [score_market(model, m, args.date, device, args.top_n) for m in args.markets]
    all_df = pd.concat([f for f in frames if f is not None and not f.empty], ignore_index=True)
    if args.out and not all_df.empty:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        all_df.to_csv(args.out, index=False)
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
