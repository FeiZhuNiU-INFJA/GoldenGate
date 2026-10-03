#!/usr/bin/env bash
# Finish remaining downloads market-by-market with crash-isolated chunks.
set -u
cd /Users/yulin/workspace/extreme_quant
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy
export EXTREME_QUANT_CN_SINA=1
# yfinance is heavily rate-limited; use akshare fallbacks (serialized).
export EQ_SKIP_YFINANCE=1
PY=/Users/yulin/miniconda3/envs/extreme_quant/bin/python
LOG=/tmp/extreme_quant_finish.log
: > "$LOG"

miss_one() {
  local m=$1
  "$PY" - <<PY
from pathlib import Path
import pandas as pd
u=pd.read_csv("dataset/$m/universe.csv")
miss=sum(1 for s in u.symbol if not (Path("dataset/$m/bars")/f"{s}.parquet").exists())
print(miss)
PY
}

download_market_chunks() {
  local market=$1
  local workers=$2
  local limit=$3
  local rounds=0
  while true; do
    rounds=$((rounds+1))
    miss=$(miss_one "$market")
    echo "[$market] round=$rounds miss=$miss $(date)" | tee -a "$LOG"
    if [[ "$miss" -eq 0 ]]; then
      echo "[$market] DONE" | tee -a "$LOG"
      return 0
    fi
    if [[ "$rounds" -gt 80 ]]; then
      echo "[$market] giving up with miss=$miss" | tee -a "$LOG"
      return 1
    fi
    MARKET="$market" WORKERS="$workers" LIMIT="$limit" "$PY" -u -B - <<'PY' >>"$LOG" 2>&1 || true
import os, sys
from pathlib import Path
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor, as_completed
sys.path.insert(0, ".")
os.environ["EXTREME_QUANT_CN_SINA"] = "1"
from data import store
from data.download import _fetch_and_write_job, download_benchmark, download_universe
from config.settings import DOWNLOAD_START_DATE

market = os.environ["MARKET"]
workers = int(os.environ["WORKERS"])
limit = int(os.environ["LIMIT"])
start = DOWNLOAD_START_DATE
end = datetime.now(timezone.utc).strftime("%Y%m%d")
download_universe(market, force=False)
download_benchmark(market, start_date=start, end_date=end, force=False)
uni = store.read_universe(market)
jobs = []
for _, row in uni.iterrows():
    sym = row["symbol"]
    if store.bar_path(market, sym).exists():
        continue
    jobs.append((market, sym, str(row["raw_symbol"]), start, end))
    if len(jobs) >= limit:
        break
print(f"{market}: jobs={len(jobs)} workers={workers}", flush=True)
if not jobs:
    raise SystemExit(0)
ok = fail = 0
with ThreadPoolExecutor(max_workers=workers) as ex:
    futs = {ex.submit(_fetch_and_write_job, j): j[1] for j in jobs}
    for fut in as_completed(futs):
        try:
            _, success, err = fut.result()
            ok += int(bool(success))
            fail += int(not success)
            if not success and err and err != "empty":
                print(f"fail {futs[fut]}: {err}", flush=True)
        except Exception as exc:
            fail += 1
            print(f"fail {futs[fut]}: {exc}", flush=True)
print(f"{market}: ok={ok} fail={fail}", flush=True)
PY
    sleep 1
  done
}

echo "=== finish start $(date) ===" | tee -a "$LOG"
echo "cn miss=$(miss_one cn) hk miss=$(miss_one hk) us miss=$(miss_one us)" | tee -a "$LOG"

download_market_chunks cn 6 100
download_market_chunks hk 1 15
download_market_chunks us 1 20

echo "=== finish end $(date) ===" | tee -a "$LOG"
echo "cn miss=$(miss_one cn) hk miss=$(miss_one hk) us miss=$(miss_one us)" | tee -a "$LOG"
