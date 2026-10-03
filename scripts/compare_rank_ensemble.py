#!/usr/bin/env python3
"""Five seeds per horizon, grades 1–20 / 21–40 / rest, then intersect.

Each seed is one model. A day's book is the symbols that land in every
seed's top 20. An empty intersection is a day with no position. Saved
ranker_*.txt checkpoints are left as they are; these models go under
checkpoints/ensemble/.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config.settings import DIR_CHECKPOINTS, DIR_DATASET, DIR_REPORTS, MARKETS
from eval.hold_returns import intersect_top_k, intersection_coverage, top_k
from train.rank_protocol import RANK_HORIZONS, REPORT_START, TUNE_START, assign_split

_spec = importlib.util.spec_from_file_location(
    "compare_rank_horizons", ROOT / "scripts" / "compare_rank_horizons.py"
)
_horizons = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_horizons)
train_market = _horizons.train_market
_bench_returns = _horizons._bench_returns
_hold_table = _horizons._hold_table
_jsonable = _horizons._jsonable

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("compare_rank_ensemble")

SEEDS = (0, 1, 2, 3, 4)
GRADE_CUTS = (20, 40)
TOP_KS = (10, 20)


def main() -> None:
    parser = argparse.ArgumentParser(description="Intersect several seeds of one horizon")
    parser.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    parser.add_argument("--grade-cuts", nargs="+", type=int, default=list(GRADE_CUTS))
    parser.add_argument("--top-ks", nargs="+", type=int, default=list(TOP_KS))
    parser.add_argument("--out-dir", default="ensemble")
    parser.add_argument("--summary", default="rank_ensemble_summary.json")
    args = parser.parse_args()
    seeds = tuple(args.seeds)
    grade_cuts = tuple(args.grade_cuts)
    top_ks = tuple(args.top_ks)

    cache = DIR_DATASET / "panels" / "rank_multi.parquet"
    panel = pd.read_parquet(cache)
    panel["trade_date"] = pd.to_datetime(panel["trade_date"]).dt.normalize()
    out_dir = DIR_CHECKPOINTS / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    pattern = str(out_dir / "ranker_{market}_h{horizon}_s{seed}.txt")
    summary = {
        "seeds": list(seeds),
        "grade_cuts": list(grade_cuts),
        "top_ks": list(top_ks),
        "markets": {},
    }
    for market in MARKETS:
        logger.info("market=%s", market)
        summary["markets"][market] = _market(
            panel.loc[panel["market"] == market].copy(),
            market,
            pattern,
            seeds,
            grade_cuts,
            top_ks,
        )
    out = DIR_REPORTS / args.summary
    out.write_text(json.dumps(_jsonable(summary), indent=2))
    _print(summary)
    print(f"summary: {out}")


def _market(
    part: pd.DataFrame,
    market: str,
    pattern: str,
    seeds: tuple[int, ...],
    grade_cuts: tuple[int, ...],
    top_ks: tuple[int, ...],
) -> dict:
    roles = assign_split(part["trade_date"], tune_start=TUNE_START, report_start=REPORT_START)
    report = part.loc[roles["report"].to_numpy()]
    returns = report[["trade_date", "symbol", *[f"ret_{h}d" for h in RANK_HORIZONS]]].copy()
    bench = _bench_returns(market, RANK_HORIZONS)
    by_seed = []
    models = []
    for seed in seeds:
        trained = train_market(
            part,
            market,
            RANK_HORIZONS,
            TUNE_START,
            REPORT_START,
            min_names=20,
            n_top=5,
            intersect_k=10,
            grade_cuts=grade_cuts,
            checkpoint_pattern=pattern.replace("{seed}", str(seed)),
            random_state=seed,
            return_scores=True,
        )
        by_seed.append(trained.pop("scores"))
        models.append(
            {
                "seed": seed,
                "models": trained["models"],
                "top5": trained["top5"],
            }
        )
    blocks = {}
    for horizon in RANK_HORIZONS:
        frames = [scores[str(horizon)] for scores in by_seed]
        one = _hold_table(top_k(frames[0], 20), returns, bench, RANK_HORIZONS, min_names=1)
        cuts = {}
        for k in top_ks:
            picks = intersect_top_k(frames, k)
            cover = intersection_coverage(frames, picks)
            cover["holds"] = _hold_table(picks, returns, bench, RANK_HORIZONS, min_names=1)
            cuts[str(k)] = cover
        blocks[str(horizon)] = {"seed0_top20": one, "intersection": cuts}
    return {"seeds": models, "books": blocks}


def _fmt(value) -> str:
    if value is None:
        return "   n/a"
    return f"{100 * value:+.2f}%"


def _print(summary: dict) -> None:
    print(f"\nIntersection of {len(summary['seeds'])} seeds, grades {summary['grade_cuts']}")
    print(f"{'market':<6} {'model':<6} {'top':<5} {'cover':>8} {'names':>6} {'hold':<6} {'days':>6} {'excess':>8}")
    for market, payload in summary["markets"].items():
        for horizon, block in payload["books"].items():
            for k, cover in block["intersection"].items():
                names = "n/a" if cover["mean_names"] is None else f"{cover['mean_names']:.2f}"
                cover_txt = "n/a" if cover["coverage"] is None else f"{100 * cover['coverage']:.1f}%"
                for hold, stats in cover["holds"].items():
                    print(
                        f"{market:<6} {horizon+'d':<6} {k:<5} {cover_txt:>8} {names:>6} "
                        f"{hold+'d':<6} {stats['days']:>6} {_fmt(stats['excess']):>8}"
                    )


if __name__ == "__main__":
    main()
