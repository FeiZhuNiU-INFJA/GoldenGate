---
name: daily-us-intersection
description: >-
  Updates S&P 500 and Nasdaq-100 daily bars through the last completed US
  close, records each market's 5-day and 10-day model top-5 intersection, and
  refreshes both recommendation trackers. Use when the user asks to update
  market data, 更新数据, 当日交集, 今天的推荐, 标普, 纳斯达克, or to refresh
  docs/live/us-intersection.md or docs/live/ndx-intersection.md.
---

# Daily S&P and Nasdaq intersection

Run this once per day for the S&P 500 and the Nasdaq-100 only. Do not update CN or HK. Report each intersection on the last **completed** US close. Do not retrain, do not rebuild `dataset/panels/rank_multi.parquet`, and do not commit unless asked.

Environment:

```bash
source "$(conda info --base)/etc/profile.d/conda.sh" && conda activate extreme_quant
```

## 1. Choose the end date

`--end` is inclusive. `_yf_history` then adds one day because yfinance's end is exclusive, so an open session becomes a partial daily bar. Never pass a session that has not closed.

US cash close is 16:00 America/New_York. If that close is still in the future, `--end` is the previous US session. Weekends stay on the last Friday. The same date is the end for both the S&P files and the Nasdaq-100 index.

## 2. Update bars

If an `update_data.py` for US is already running, wait for it. Do not start a second one. Set `EQ_SKIP_YFINANCE=1`. Do not raise `--workers` above 8.

```bash
EQ_SKIP_YFINANCE=1 PYTHONUNBUFFERED=1 python scripts/update_data.py --markets us ndx --end YYYYMMDD
```

Nasdaq-100 names that are also in the S&P 500 share `dataset/us/bars/` and are updated by the `us` market. `ndx` updates the Nasdaq-100 index and the names that live only under `dataset/ndx/bars/`.

One yfinance warning on the S&P benchmark is fine when the log then says it tried Sina and the row count moved. A symbol with `fail=0` whose last bar is months behind the benchmark no longer trades.

## 3. Check the last close

Confirm the S&P benchmark, the Nasdaq-100 benchmark, and the liquid names stop on `--end`, and that the next session is absent. `SYN*` files are synthetic and stay old.

## 4. Refresh both trackers

```bash
python scripts/update_us_recommendations.py
```

This rewrites both ledgers and their pages:

- S&P 500: `docs/live/us-intersection.md` and `docs/live/us-intersection.html`, models `checkpoints/ranker_us_h5.txt` and `checkpoints/ranker_us_h10.txt`. First date is 2026-10-01. A date with fewer than 450 names is left unrecorded.
- Nasdaq-100: `docs/live/ndx-intersection.md` and `docs/live/ndx-intersection.html`, models `checkpoints/ranker_ndx_h5.txt` and `checkpoints/ranker_ndx_h10.txt`. First date is 2026-10-02. A date with fewer than 80 names is left unrecorded.

Shared rules, do not redo them by hand:

- Score sign is `+1`. Each model contributes its top 5. The recommendation is the intersection. An empty intersection is 空仓.
- Do not edit the JSON to force a thin day in.
- Dates before each book's first date are not backfilled.
- The HTML is the performance view. Do not rebuild it by hand.

## 5. Reply

Lead with each book's newest signal date, the names (or 空仓), and any new cumulative row: 持有天数, 组合, 大盘, 超额. S&P excess is versus the S&P 500. Nasdaq excess is versus the Nasdaq-100. Say which of 5 / 10 / 20 日 are still 未到期. Point at both HTML pages. The signal date is the last completed US session, not today's calendar date while the US cash session is open.
