#!/usr/bin/env python3
"""Train 5d/10d/20d cross-sectional rankers and compare top-5 holding returns.

Every date before 2024-06-01 is training data for the saved model. The number
of trees and the score sign are chosen on 2024-01-01 .. 2024-05-31, then the
model is refit on the full pre-report sample. 2024-06-01 onward is only scored.

The 20-day model is trained again here; existing h20 checkpoints are replaced.
"""
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

from config.settings import DIR_CHECKPOINTS, DIR_DATASET, DIR_REPORTS, MARKETS
from data import store
from eval.hold_returns import book_daily, intersect_top_k, intersection_coverage, summarize_hold
from eval.hold_returns import top_k as select_top_k
from eval.rank_metrics import summarize_scores
from labels.cross_section import GRADE_CUTS, LABEL_GAIN, assign_cross_section
from train.rank_features import RANK_FEATURES, rank_frame
from train.rank_protocol import RANK_HORIZONS, REPORT_START, TUNE_START, assign_split

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("compare_rank_horizons")

READ_COLS = [
    "trade_date",
    "symbol",
    "market",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "amount",
    "bench_close",
]
MIN_CHILD = {"cn": 200, "hk": 20, "us": 50, "ndx": 50}


def _groups(dates: pd.Series) -> np.ndarray:
    return dates.groupby(dates, sort=False).size().to_numpy()


def build_panel(markets: list[str], horizons: tuple[int, ...]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    chunks: list[pd.DataFrame] = []
    for market in markets:
        if market == "ndx":
            chunks.append(_ndx_panel(horizons))
            continue
        symbols = store.list_labeled_symbols(market)
        logger.info("%s labeled symbols=%s", market, len(symbols))
        for sym in tqdm(symbols, desc=f"panel-{market}"):
            df = store.read_parquet(store.labeled_path(market, sym))
            if df is None or df.empty or "bench_close" not in df.columns:
                continue
            keep = [c for c in READ_COLS if c in df.columns]
            feat = rank_frame(df[keep], horizons=horizons)
            feat = feat.dropna(subset=list(RANK_FEATURES))
            if feat.empty:
                continue
            frames.append(feat)
            if len(frames) >= 400:
                chunks.append(pd.concat(frames, ignore_index=True))
                frames = []
    if frames:
        chunks.append(pd.concat(frames, ignore_index=True))
    if not chunks:
        raise RuntimeError("no rows for rank panel")
    panel = pd.concat(chunks, ignore_index=True)
    panel["trade_date"] = pd.to_datetime(panel["trade_date"]).dt.normalize()
    return panel


def _ndx_panel(horizons: tuple[int, ...]) -> pd.DataFrame:
    """Nasdaq-100 bars, with excess versus the Nasdaq-100 index.

    Names that also sit in the S&P 500 are read from ``dataset/us/bars``.
    Their labeled files carry the S&P benchmark, so they are not reused here.
    """
    bench = store.read_parquet(store.benchmark_path("ndx"))
    if bench is None or bench.empty:
        raise RuntimeError("missing Nasdaq-100 benchmark")
    bench = bench.copy()
    bench["trade_date"] = pd.to_datetime(bench["trade_date"]).dt.normalize()
    bench_close = bench.drop_duplicates("trade_date").set_index("trade_date")["close"]
    symbols = store.list_bar_symbols("ndx")
    logger.info("ndx bar symbols=%s", len(symbols))
    frames: list[pd.DataFrame] = []
    for sym in tqdm(symbols, desc="panel-ndx"):
        path = store.resolve_bar_path("ndx", sym)
        raw = store.read_parquet(path) if path is not None else None
        if raw is None or raw.empty:
            continue
        raw = raw.copy()
        raw["trade_date"] = pd.to_datetime(raw["trade_date"]).dt.normalize()
        raw["symbol"] = sym
        raw["market"] = "ndx"
        raw["bench_close"] = raw["trade_date"].map(bench_close)
        feat = rank_frame(raw, horizons=horizons)
        feat = feat.dropna(subset=list(RANK_FEATURES))
        if not feat.empty:
            frames.append(feat)
    if not frames:
        raise RuntimeError("no nasdaq rows")
    return pd.concat(frames, ignore_index=True)


def _fit_ranker(
    train: pd.DataFrame,
    val: pd.DataFrame | None,
    min_child: int,
    n_estimators: int,
    features: tuple[str, ...] | list[str] | None = None,
    random_state: int = 0,
):
    import lightgbm as lgb

    cols = list(features or RANK_FEATURES)
    train = train.sort_values(["trade_date", "symbol"])
    model = lgb.LGBMRanker(
        objective="lambdarank",
        metric="ndcg",
        ndcg_eval_at=[5],
        learning_rate=0.05,
        num_leaves=63,
        min_child_samples=min_child,
        n_estimators=n_estimators,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        label_gain=LABEL_GAIN,
        n_jobs=8,
        random_state=random_state,
        verbosity=-1,
    )
    kwargs = {}
    if val is not None and not val.empty:
        val = val.sort_values(["trade_date", "symbol"])
        kwargs = {
            "eval_set": [(val[cols], val["relevance"])],
            "eval_group": [_groups(val["trade_date"])],
            "callbacks": [lgb.early_stopping(40), lgb.log_evaluation(50)],
        }
    model.fit(
        train[cols],
        train["relevance"],
        group=_groups(train["trade_date"]),
        **kwargs,
    )
    return model


def _tree_count(model, cap: int) -> int:
    best = int(getattr(model, "best_iteration_", -1) or -1)
    if best < 1:
        return cap
    return best


def _score(
    model,
    df: pd.DataFrame,
    sign: int,
    features: tuple[str, ...] | list[str] | None = None,
) -> pd.DataFrame:
    cols = list(features or RANK_FEATURES)
    out = df.sort_values(["trade_date", "symbol"]).copy()
    out["score"] = model.predict(out[cols]) * sign
    return out


def _jsonable(value):
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if np.isfinite(number) else None
    if isinstance(value, (np.integer, int)):
        return int(value)
    if value is None or isinstance(value, str):
        return value
    return value


def _bench_returns(
    market: str,
    horizons: tuple[int, ...],
    benchmark: pd.DataFrame | None = None,
) -> pd.DataFrame:
    bench = benchmark if benchmark is not None else store.read_parquet(store.benchmark_path(market))
    if bench is None or bench.empty:
        raise FileNotFoundError(f"missing benchmark for {market}")
    df = bench.sort_values("trade_date").drop_duplicates("trade_date")
    close = df["close"].astype(np.float64)
    out = pd.DataFrame({"trade_date": pd.to_datetime(df["trade_date"]).dt.normalize()})
    for h in horizons:
        future = close.shift(-h)
        ret = (future / close - 1.0).mask((close <= 0) | (future <= 0))
        out[f"bench_ret_{h}d"] = ret.to_numpy()
    return out


def _hold_table(picks: pd.DataFrame, returns: pd.DataFrame, bench: pd.DataFrame, horizons: tuple[int, ...], min_names: int) -> dict:
    table = {}
    for h in horizons:
        series = bench.set_index("trade_date")[f"bench_ret_{h}d"]
        daily = book_daily(picks, returns, series, ret_col=f"ret_{h}d", min_names=min_names)
        table[str(h)] = summarize_hold(daily, step=h)
    return table


def train_market(
    part: pd.DataFrame,
    market: str,
    horizons: tuple[int, ...],
    tune_start: str,
    report_start: str,
    min_names: int,
    n_top: int,
    intersect_k: int,
    grade_cuts: tuple[int, ...] = GRADE_CUTS,
    features: tuple[str, ...] | list[str] | None = None,
    checkpoint_pattern: str = "ranker_{market}_h{horizon}.txt",
    benchmark: pd.DataFrame | None = None,
    random_state: int = 0,
    return_scores: bool = False,
) -> dict:
    roles = assign_split(part["trade_date"], tune_start=tune_start, report_start=report_start)
    part = part.reset_index(drop=True)
    roles = roles.reset_index(drop=True)
    report = part.loc[roles["report"]].copy()
    bench = _bench_returns(market, horizons, benchmark=benchmark)
    ret_cols = [f"ret_{h}d" for h in horizons]
    returns = report[["trade_date", "symbol", *ret_cols]].copy()
    cols = list(features or RANK_FEATURES)
    scored: dict[int, pd.DataFrame] = {}
    models_report = {}
    min_child = MIN_CHILD.get(market, 50)

    for horizon in horizons:
        label_col = f"exret_{horizon}d"
        train_rows = part.loc[roles["train"]].dropna(subset=[label_col])
        labeled = assign_cross_section(
            train_rows,
            value_col=label_col,
            min_names=min_names,
            grade_cuts=grade_cuts,
        )
        labeled_roles = assign_split(labeled["trade_date"], tune_start=tune_start, report_start=report_start)
        fit = labeled.loc[labeled_roles["fit"].to_numpy()]
        tune = labeled.loc[labeled_roles["tune"].to_numpy()]
        if fit.empty or tune.empty:
            raise RuntimeError(f"{market} h{horizon} missing fit or tune rows")
        logger.info("%s h%s fit=%s tune=%s report=%s", market, horizon, len(fit), len(tune), len(report))
        selector = _fit_ranker(
            fit, tune, min_child, n_estimators=400, features=cols, random_state=random_state
        )
        n_trees = _tree_count(selector, cap=400)
        tune_scored = _score(selector, tune, sign=1, features=cols)
        tune_stats = summarize_scores(tune_scored, label_col=label_col, step=horizon)
        sign = -1 if tune_stats["ic"] is not None and tune_stats["ic"] < 0 else 1
        if sign < 0:
            logger.info("%s h%s tune IC negative; freezing score sign at -1", market, horizon)
        logger.info("%s h%s refit trees=%s sign=%s", market, horizon, n_trees, sign)
        final = _fit_ranker(
            labeled, None, min_child, n_estimators=n_trees, features=cols, random_state=random_state
        )
        path = DIR_CHECKPOINTS / checkpoint_pattern.format(market=market, horizon=horizon)
        final.booster_.save_model(str(path))
        scored[horizon] = _score(final, report, sign=sign, features=cols)
        models_report[str(horizon)] = {
            "checkpoint": str(path),
            "trees": n_trees,
            "sign": sign,
            "tune_ic": tune_stats["ic"],
            "grade_cuts": list(grade_cuts),
            "label_gain": list(LABEL_GAIN),
        }

    top5 = {}
    for horizon, frame in scored.items():
        chosen = select_top_k(frame, n_top)
        top5[str(horizon)] = _hold_table(chosen, returns, bench, horizons, min_names=n_top)

    frames = [scored[h] for h in horizons]
    intersection = intersect_top_k(frames, intersect_k)
    cover = intersection_coverage(frames, intersection)
    cover["holds"] = _hold_table(intersection, returns, bench, horizons, min_names=1)
    result = {"models": models_report, "top5": top5, "intersection": cover}
    if return_scores:
        result["scores"] = {
            str(horizon): frame[["trade_date", "symbol", "score"]].copy()
            for horizon, frame in scored.items()
        }
    return result


def _fmt_pct(value) -> str:
    if value is None:
        return "  n/a"
    return f"{100 * value:+.2f}%"


def _print_tables(summary: dict) -> None:
    print("\nTop 5 equal-weight close-to-close return")
    header = f"{'market':<6} {'model':<6} {'hold':<6} {'days':>6} {'port':>8} {'bench':>8} {'excess':>8} {'ex_step':>10}"
    print(header)
    for market, payload in summary["markets"].items():
        for model, holds in payload["top5"].items():
            for hold, stats in holds.items():
                print(
                    f"{market:<6} {model+'d':<6} {hold+'d':<6} {stats['days']:>6} "
                    f"{_fmt_pct(stats['port']):>8} {_fmt_pct(stats['bench']):>8} "
                    f"{_fmt_pct(stats['excess']):>8} {_fmt_pct(stats['excess_nonoverlap']):>10}"
                )
    print("\nIntersection of each model's top 10")
    print(f"{'market':<6} {'cover':>8} {'names':>6} {'hold':<6} {'days':>6} {'port':>8} {'bench':>8} {'excess':>8}")
    for market, payload in summary["markets"].items():
        block = payload["intersection"]
        names = "n/a" if block["mean_names"] is None else f"{block['mean_names']:.2f}"
        cover = "n/a" if block["coverage"] is None else f"{100 * block['coverage']:.1f}%"
        for hold, stats in block["holds"].items():
            print(
                f"{market:<6} {cover:>8} {names:>6} {hold+'d':<6} {stats['days']:>6} "
                f"{_fmt_pct(stats['port']):>8} {_fmt_pct(stats['bench']):>8} {_fmt_pct(stats['excess']):>8}"
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare 5/10/20-day cross-sectional rankers")
    parser.add_argument("--markets", nargs="+", default=list(MARKETS), choices=list(MARKETS))
    parser.add_argument("--horizons", nargs="+", type=int, default=list(RANK_HORIZONS))
    parser.add_argument("--tune-start", default=TUNE_START)
    parser.add_argument("--report-start", default=REPORT_START)
    parser.add_argument("--min-names", type=int, default=20)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--intersect-k", type=int, default=10)
    parser.add_argument("--rebuild-panel", action="store_true")
    args = parser.parse_args()
    horizons = tuple(args.horizons)

    cache = DIR_DATASET / "panels" / "rank_multi.parquet"
    if cache.exists() and not args.rebuild_panel:
        logger.info("loading panel %s", cache)
        panel = pd.read_parquet(cache)
        panel["trade_date"] = pd.to_datetime(panel["trade_date"]).dt.normalize()
        missing = [market for market in args.markets if market not in set(panel["market"])]
        if missing:
            extra = build_panel(missing, horizons)
            panel = pd.concat([panel, extra], ignore_index=True)
            panel.to_parquet(cache, index=False)
            logger.info("appended %s to %s rows=%s", missing, cache, len(panel))
        panel = panel.loc[panel["market"].isin(args.markets)].reset_index(drop=True)
    else:
        panel = build_panel(args.markets, horizons)
        cache.parent.mkdir(parents=True, exist_ok=True)
        panel.to_parquet(cache, index=False)
        logger.info("wrote panel %s rows=%s", cache, len(panel))

    needed = [*RANK_FEATURES, *[f"ret_{h}d" for h in horizons], *[f"exret_{h}d" for h in horizons]]
    missing = [c for c in needed if c not in panel.columns]
    if missing:
        raise RuntimeError(f"panel missing {missing}; rerun with --rebuild-panel")

    summary = {
        "tune_start": args.tune_start,
        "report_start": args.report_start,
        "top_k": args.top_k,
        "grade_cuts": list(GRADE_CUTS),
        "label_gain": list(LABEL_GAIN),
        "intersect_k": args.intersect_k,
        "horizons": list(horizons),
        "markets": {},
    }
    DIR_CHECKPOINTS.mkdir(parents=True, exist_ok=True)
    out = DIR_REPORTS / "rank_horizons_summary.json"
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
        )
        out.write_text(json.dumps(_jsonable(summary), indent=2))
        logger.info("wrote partial summary %s", out)
    _print_tables(summary)
    print(f"\nsummary: {out}")


if __name__ == "__main__":
    main()
