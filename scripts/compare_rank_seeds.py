#!/usr/bin/env python3
"""Refit the three-grade rankers on several seeds and compare top-5 excess.

Checkpoints go to a temporary directory. The saved ranker_*.txt models stay
on seed 0.
"""
from __future__ import annotations

import importlib.util
import json
import logging
import sys
import tempfile
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config.settings import DIR_DATASET, DIR_REPORTS, MARKETS
from train.rank_protocol import RANK_HORIZONS, REPORT_START, TUNE_START

_spec = importlib.util.spec_from_file_location(
    "compare_rank_horizons", ROOT / "scripts" / "compare_rank_horizons.py"
)
_horizons = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_horizons)
train_market = _horizons.train_market
_jsonable = _horizons._jsonable

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("compare_rank_seeds")

SEEDS = (0, 1, 2, 3, 4)


def main() -> None:
    cache = DIR_DATASET / "panels" / "rank_multi.parquet"
    panel = pd.read_parquet(cache)
    panel["trade_date"] = pd.to_datetime(panel["trade_date"]).dt.normalize()
    rows = []
    with tempfile.TemporaryDirectory(prefix="rank_seeds_") as tmp:
        pattern = str(Path(tmp) / "ranker_{market}_h{horizon}.txt")
        for seed in SEEDS:
            for market in MARKETS:
                logger.info("seed=%s market=%s", seed, market)
                trained = train_market(
                    panel.loc[panel["market"] == market].copy(),
                    market,
                    RANK_HORIZONS,
                    TUNE_START,
                    REPORT_START,
                    min_names=20,
                    n_top=5,
                    intersect_k=10,
                    checkpoint_pattern=pattern,
                    random_state=seed,
                )
                for horizon, model in trained["models"].items():
                    holds = trained["top5"][horizon]
                    rows.append(
                        {
                            "seed": seed,
                            "market": market,
                            "horizon": int(horizon),
                            "trees": model["trees"],
                            "sign": model["sign"],
                            "tune_ic": model["tune_ic"],
                            "excess": {hold: holds[hold]["excess"] for hold in holds},
                        }
                    )
    out = DIR_REPORTS / "rank_seed_stability.json"
    out.write_text(json.dumps(_jsonable(rows), indent=2))
    _print(rows)
    print(f"summary: {out}")


def _print(rows: list[dict]) -> None:
    print(f"{'market':<6} {'model':<6} {'hold':<6} {'min':>8} {'median':>8} {'max':>8} {'signs'}")
    frame = pd.DataFrame(rows)
    for market in MARKETS:
        for horizon in RANK_HORIZONS:
            part = frame[(frame["market"] == market) & (frame["horizon"] == horizon)]
            ordered = part.sort_values("seed")
            signs = " ".join(f"{int(s):+d}" for s in ordered["sign"])
            trees = " ".join(str(int(t)) for t in ordered["trees"])
            print(f"{market} {horizon}d  signs {signs}  trees {trees}")
            holds = sorted({h for item in part["excess"] for h in item}, key=int)
            for hold in holds:
                values = sorted(float(item[hold]) for item in part["excess"])
                mid = values[len(values) // 2]
                print(
                    f"  hold {hold}d  min {100 * values[0]:+.2f}%  "
                    f"median {100 * mid:+.2f}%  max {100 * values[-1]:+.2f}%"
                )


if __name__ == "__main__":
    main()
