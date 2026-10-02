#!/usr/bin/env bash
# One Python process per small batch so a crash cannot wipe the whole run.
set -u
cd /Users/yulin/workspace/extreme_quant
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy
export EXTREME_QUANT_CN_SINA=1
PY=/Users/yulin/miniconda3/envs/extreme_quant/bin/python
LOG=/tmp/extreme_quant_batch.log
: > "$LOG"

miss_count() {
  "$PY" - <<'PY'
from pathlib import Path
import pandas as pd
root=Path("dataset")
miss=0
for m in ["cn","hk","us"]:
    u=pd.read_csv(root/m/"universe.csv")
    miss += sum(1 for s in u.symbol if not (root/m/"bars"/f"{s}.parquet").exists())
print(miss)
PY
}

echo "start miss=$(miss_count)" | tee -a "$LOG"

for round in $(seq 1 50); do
  miss=$(miss_count)
  echo "===== round $round miss=$miss $(date) =====" | tee -a "$LOG"
  if [[ "$miss" -eq 0 ]]; then
    echo DONE | tee -a "$LOG"
    exit 0
  fi
  # Download at most 120 symbols in this short-lived process, then exit cleanly.
  "$PY" -u -B - <<'PY' >>"$LOG" 2>&1 || true
import os, sys
from pathlib import Path
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor, as_completed
sys.path.insert(0, ".")
os.environ["EXTREME_QUANT_CN_SINA"] = "1"
from data import store
from data.download import _fetch_and_write_job, download_benchmark, download_universe
from config.settings import DOWNLOAD_START_DATE

start = DOWNLOAD_START_DATE
end = datetime.now(timezone.utc).strftime("%Y%m%d")
LIMIT = 120
WORKERS = 6

for market in ["cn", "hk", "us"]:
    download_universe(market, force=False)
    download_benchmark(market, start_date=start, end_date=end, force=False)
    uni = store.read_universe(market)
    jobs = []
    for _, row in uni.iterrows():
        sym = row["symbol"]
        if store.bar_path(market, sym).exists():
            continue
        jobs.append((market, sym, str(row["raw_symbol"]), start, end))
        if len(jobs) >= LIMIT:
            break
    if not jobs:
        print(f"{market}: nothing", flush=True)
        continue
    workers = WORKERS if market == "cn" else 3
    ok = fail = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(_fetch_and_write_job, j): j[1] for j in jobs}
        for fut in as_completed(futs):
            try:
                _, success, err = fut.result()
                ok += int(success)
                fail += int(not success)
                if not success and err and err != "empty":
                    print(f"fail {futs[fut]}: {err}", flush=True)
            except Exception as exc:
                fail += 1
                print(f"fail {futs[fut]}: {exc}", flush=True)
    print(f"{market}: attempted={len(jobs)} ok={ok} fail={fail}", flush=True)
PY
  sleep 1
done

echo "final miss=$(miss_count)" | tee -a "$LOG"
exit 1
