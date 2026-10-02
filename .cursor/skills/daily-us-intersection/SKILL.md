---
name: daily-us-intersection
description: >-
  Updates CN, HK, and US daily bars through the last completed close, records
  the US 5-day and 10-day model top-5 intersection, and refreshes the
  recommendation tracker. Use when the user asks to update market data, 更新数据,
  当日交集, 今天的推荐, or to refresh docs/live/us-intersection.md.
---

# Daily US intersection

Run this once per day. Report the intersection on the last **completed** US close. Do not retrain, do not rebuild `dataset/panels/rank_multi.parquet`, and do not commit unless asked.

Environment:

```bash
source "$(conda info --base)/etc/profile.d/conda.sh" && conda activate extreme_quant
```

## 1. Choose end dates

`--end` is inclusive. `_yf_history` then adds one day because yfinance's end is exclusive, so an open session becomes a partial daily bar. Never pass a session that has not closed.

- US cash close is 16:00 America/New_York. If that close is still in the future, US `--end` is the previous US session. Weekends stay on the last Friday.
- CN closes 15:00 Asia/Shanghai, HK closes 16:00 Asia/Hong_Kong. After that, `--end` is today. A holiday that adds `+0` benchmark rows is a finished close on the previous session, not a failed download.

## 2. Update bars

If an `update_data.py` for the same market is already running, wait for it. Do not start a second one.

Run US and CN/HK in parallel. Set `EQ_SKIP_YFINANCE=1` on both so US and HK go to Sina. yfinance rate-limits a full universe; do not raise `--workers` above 8.

```bash
EQ_SKIP_YFINANCE=1 PYTHONUNBUFFERED=1 python scripts/update_data.py --markets us --end YYYYMMDD
EQ_SKIP_YFINANCE=1 PYTHONUNBUFFERED=1 python scripts/update_data.py --markets cn hk --end YYYYMMDD
```

One yfinance warning on the US benchmark is fine when the log then says it tried Sina and the row count moved. A symbol with `fail=0` whose last bar is months behind the benchmark no longer trades. A CN `fail` after a stalled worker can still have been written by the Sina retry; check that symbol's max date before relaunching the market.

## 3. Check the last close

Confirm the US benchmark and the liquid names stop on the US `--end`, and that the next session is absent. CN and HK may stop earlier on a holiday. `SYN*` files are synthetic and stay old.

## 4. Refresh the tracker

```bash
python scripts/update_us_recommendations.py
```

This scores any US session after the first date in `docs/live/us-intersection.json` and rewrites `docs/live/us-intersection.md`. Rules already in the script, do not redo them by hand:

- Models are `checkpoints/ranker_us_h5.txt` and `checkpoints/ranker_us_h10.txt`, score sign `+1`, top 5 each, intersection only.
- A date with fewer than 450 names is left unrecorded. Finish the bars and rerun. Do not edit the JSON to force it in.
- An empty intersection is recorded as 空仓.
- Dates before the first ledger entry are not backfilled.

## 5. Reply

Lead with the newest signal date, the names (or 空仓), and any new cumulative row: 持有天数, 组合, 大盘, 超额. Then the last close of CN, HK, and US. Say which of 5 / 10 / 20 日 are still 未到期. The signal date is the last completed US session, not today's calendar date while the US cash session is open.
