#!/usr/bin/env python3
"""Cross-sectional 20d ranker: z-scored excess return, LightGBM LambdaRank."""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config.settings import DIR_CHECKPOINTS, DIR_DATASET, DIR_REPORTS, MARKETS, TEST_END, TEST_START, TRAIN_END, TRAIN_START, VAL_END, VAL_START
from data import store
from eval.rank_metrics import summarize_scores
from labels.cross_section import assign_cross_section
from train.rank_features import RANK_FEATURES, features_from_labeled

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("train_ranker")

SPLITS = {
    "train": (TRAIN_START, TRAIN_END),
    "val": (VAL_START, VAL_END),
    "test": (TEST_START, TEST_END),
}


def _split_name(dates: pd.Series) -> pd.Series:
    out = pd.Series("drop", index=dates.index)
    for name, (start, end) in SPLITS.items():
        mask = (dates >= pd.Timestamp(start)) & (dates <= pd.Timestamp(end))
        out = out.mask(mask, name)
    return out


def build_panel(markets: list[str], horizon: int, min_names: int) -> pd.DataFrame:
    label_col = f"exret_{horizon}d"
    frames: list[pd.DataFrame] = []
    for market in markets:
        symbols = store.list_labeled_symbols(market)
        logger.info("%s labeled symbols=%s", market, len(symbols))
        for sym in tqdm(symbols, desc=f"panel-{market}"):
            df = store.read_parquet(store.labeled_path(market, sym))
            if df is None or df.empty or label_col not in df.columns:
                continue
            feat = features_from_labeled(df, horizon=horizon)
            feat = feat.dropna(subset=[*RANK_FEATURES, label_col])
            if not feat.empty:
                frames.append(feat)
    if not frames:
        raise RuntimeError("no rows for rank panel")
    panel = pd.concat(frames, ignore_index=True)
    panel["trade_date"] = pd.to_datetime(panel["trade_date"])
    panel = assign_cross_section(panel, value_col=label_col, min_names=min_names)
    panel["split"] = _split_name(panel["trade_date"])
    panel = panel.loc[panel["split"] != "drop"].reset_index(drop=True)
    return panel


def _groups(dates: pd.Series) -> np.ndarray:
    return dates.groupby(dates, sort=False).size().to_numpy()


def _fit_market(train: pd.DataFrame, val: pd.DataFrame, min_child: int):
    import lightgbm as lgb

    train = train.sort_values(["trade_date", "symbol"])
    val = val.sort_values(["trade_date", "symbol"])
    model = lgb.LGBMRanker(
        objective="lambdarank",
        metric="ndcg",
        ndcg_eval_at=[5],
        learning_rate=0.05,
        num_leaves=63,
        min_child_samples=min_child,
        n_estimators=400,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        n_jobs=8,
        random_state=0,
        verbosity=-1,
    )
    model.fit(
        train[list(RANK_FEATURES)],
        train["relevance"],
        group=_groups(train["trade_date"]),
        eval_set=[(val[list(RANK_FEATURES)], val["relevance"])],
        eval_group=[_groups(val["trade_date"])],
        callbacks=[lgb.early_stopping(40), lgb.log_evaluation(50)],
    )
    return model


def _score(model, df: pd.DataFrame) -> pd.DataFrame:
    out = df.sort_values(["trade_date", "symbol"]).copy()
    out["score"] = model.predict(out[list(RANK_FEATURES)])
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="Train a daily cross-sectional ranker")
    p.add_argument("--markets", nargs="+", default=list(MARKETS), choices=list(MARKETS))
    p.add_argument("--horizon", type=int, default=20)
    p.add_argument("--min-names", type=int, default=20)
    args = p.parse_args()

    cache = DIR_DATASET / "panels" / f"rank{args.horizon}.parquet"
    if cache.exists():
        logger.info("loading panel %s", cache)
        panel = pd.read_parquet(cache)
        panel["trade_date"] = pd.to_datetime(panel["trade_date"])
        panel = panel.loc[panel["market"].isin(args.markets)].reset_index(drop=True)
    else:
        panel = build_panel(args.markets, args.horizon, args.min_names)
        cache.parent.mkdir(parents=True, exist_ok=True)
        panel.to_parquet(cache, index=False)
        logger.info("wrote panel %s rows=%s", cache, len(panel))

    label_col = f"exret_{args.horizon}d"
    summary = {"horizon": args.horizon, "features": list(RANK_FEATURES), "markets": {}}
    min_child = {"cn": 200, "hk": 20, "us": 50}
    DIR_CHECKPOINTS.mkdir(parents=True, exist_ok=True)

    for market in args.markets:
        part = panel.loc[panel["market"] == market]
        train = part.loc[part["split"] == "train"]
        val = part.loc[part["split"] == "val"]
        test = part.loc[part["split"] == "test"]
        if train.empty or val.empty:
            logger.warning("%s missing train/val rows", market)
            continue
        logger.info("%s rows train=%s val=%s test=%s", market, len(train), len(val), len(test))
        model = _fit_market(train, val, min_child.get(market, 50))
        path = DIR_CHECKPOINTS / f"ranker_{market}_h{args.horizon}.txt"
        model.booster_.save_model(str(path))
        market_report = {"checkpoint": str(path), "best_iteration": int(model.best_iteration_ or 0), "sign": 1}
        scored_val = _score(model, val)
        val_stats = summarize_scores(scored_val, label_col=label_col, step=args.horizon)
        if val_stats["ic"] is not None and val_stats["ic"] < 0:
            market_report["sign"] = -1
            logger.info("%s val IC negative; flipping score sign", market)
        for split_name, frame in ("val", val), ("test", test):
            if frame.empty:
                continue
            scored = scored_val if split_name == "val" else _score(model, frame)
            if market_report["sign"] < 0:
                scored = scored.copy()
                scored["score"] = -scored["score"]
            stats = summarize_scores(scored, label_col=label_col, step=args.horizon)
            market_report[split_name] = stats
            logger.info("%s %s %s", market, split_name, stats)
        summary["markets"][market] = market_report

    out = DIR_REPORTS / f"rank{args.horizon}_summary.json"
    out.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"summary: {out}")


if __name__ == "__main__":
    main()
