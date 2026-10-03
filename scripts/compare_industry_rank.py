#!/usr/bin/env python3
"""Rankers with the existing price features plus industry-relative factors.

Labels, the train/validation split, and the top-5 return table match
scripts/compare_rank_horizons.py. Checkpoints are ranker_{market}_h{horizon}_ind.txt
so the live 5-day and 10-day models are left in place.
"""
from __future__ import annotations

import importlib.util
import json
import logging
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config.settings import DIR_DATASET, DIR_REPORTS, MARKETS

_spec = importlib.util.spec_from_file_location(
    "compare_rank_horizons", ROOT / "scripts" / "compare_rank_horizons.py"
)
_horizons = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_horizons)
_jsonable = _horizons._jsonable
_print_tables = _horizons._print_tables
train_market = _horizons.train_market
from train.industry_features import INDUSTRY_FEATURES, add_industry_relative
from train.rank_features import RANK_FEATURES
from train.rank_protocol import RANK_HORIZONS, REPORT_START, TUNE_START

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("compare_industry_rank")

FEATURES = tuple(RANK_FEATURES) + INDUSTRY_FEATURES


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Compare rankers with industry-relative factors")
    parser.add_argument("--markets", nargs="+", default=list(MARKETS), choices=list(MARKETS))
    parser.add_argument("--horizons", nargs="+", type=int, default=list(RANK_HORIZONS))
    parser.add_argument("--tune-start", default=TUNE_START)
    parser.add_argument("--report-start", default=REPORT_START)
    parser.add_argument("--min-names", type=int, default=20)
    parser.add_argument("--min-peers", type=int, default=5)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--intersect-k", type=int, default=10)
    args = parser.parse_args()
    horizons = tuple(args.horizons)

    panel = _panel(args.markets, args.min_peers)
    summary = {
        "features": list(FEATURES),
        "min_peers": args.min_peers,
        "tune_start": args.tune_start,
        "report_start": args.report_start,
        "top_k": args.top_k,
        "intersect_k": args.intersect_k,
        "horizons": list(horizons),
        "industry_coverage": _coverage(panel),
        "markets": {},
    }
    out = DIR_REPORTS / "rank_industry_summary.json"
    for market in args.markets:
        part = panel.loc[panel["market"] == market].copy()
        summary["markets"][market] = train_market(
            part,
            market,
            horizons,
            args.tune_start,
            args.report_start,
            args.min_names,
            args.top_k,
            args.intersect_k,
            features=FEATURES,
            checkpoint_pattern="ranker_{market}_h{horizon}_ind.txt",
        )
        out.write_text(json.dumps(_jsonable(summary), indent=2))
        logger.info("wrote partial summary %s", out)
    _print_tables(summary)
    _print_baseline_gap(summary)
    print(f"\nsummary: {out}")


def _panel(markets: list[str], min_peers: int) -> pd.DataFrame:
    cache = DIR_DATASET / "panels" / "rank_multi.parquet"
    if not cache.exists():
        raise RuntimeError(f"missing {cache}; run scripts/compare_rank_horizons.py --rebuild-panel first")
    panel = pd.read_parquet(cache)
    panel["trade_date"] = pd.to_datetime(panel["trade_date"]).dt.normalize()
    panel = panel.loc[panel["market"].isin(markets)].reset_index(drop=True)
    labels = pd.concat([_industry(market) for market in markets], ignore_index=True)
    enriched = add_industry_relative(panel, labels, min_peers=min_peers)
    missing = [col for col in FEATURES if enriched[col].isna().all()]
    if missing:
        raise RuntimeError(f"industry features are empty: {missing}")
    logger.info(
        "panel rows=%s industry-peer rows=%s",
        len(enriched),
        int((enriched["industry_peers"] >= min_peers).sum()),
    )
    return enriched


def _industry(market: str) -> pd.DataFrame:
    path = DIR_DATASET / market / "industry.parquet"
    if not path.exists():
        raise RuntimeError(f"missing {path}; run scripts/build_industry_map.py")
    frame = pd.read_parquet(path, columns=["symbol", "industry"])
    frame["market"] = market
    return frame


def _coverage(panel: pd.DataFrame) -> dict:
    out = {}
    for market, part in panel.groupby("market"):
        out[str(market)] = {
            "rows": int(len(part)),
            "peer_rows": int((part["industry_peers"] > 0).sum()),
            "peer_share": float((part["industry_peers"] > 0).mean()),
        }
    return out


def _print_baseline_gap(summary: dict) -> None:
    path = DIR_REPORTS / "rank_horizons_summary.json"
    if not path.exists():
        return
    baseline = json.loads(path.read_text())
    print("\nExcess versus the 7 price features (same hold as the model)")
    print(f"{'market':<6} {'model':<6} {'base':>8} {'industry':>8} {'delta':>8}")
    for market, payload in summary["markets"].items():
        base_market = baseline.get("markets", {}).get(market, {})
        for horizon, holds in payload["top5"].items():
            base = base_market.get("top5", {}).get(horizon, {}).get(horizon, {}).get("excess")
            new = holds.get(horizon, {}).get("excess")
            delta = None if base is None or new is None else new - base
            print(
                f"{market:<6} {horizon+'d':<6} {_pct(base):>8} {_pct(new):>8} {_pct(delta):>8}"
            )


def _pct(value) -> str:
    if value is None:
        return "n/a"
    return f"{100 * value:+.2f}%"


if __name__ == "__main__":
    main()
