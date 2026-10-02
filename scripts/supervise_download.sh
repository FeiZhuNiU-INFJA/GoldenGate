#!/usr/bin/env bash
# Keep restarting resume_download until coverage is complete or max attempts.
set -u
cd /Users/yulin/workspace/extreme_quant
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy
export EXTREME_QUANT_CN_SINA=1
PY=/Users/yulin/miniconda3/envs/extreme_quant/bin/python
LOG=/tmp/extreme_quant_download.log
: > "$LOG"

coverage() {
  "$PY" - <<'PY'
from pathlib import Path
import pandas as pd
root = Path("dataset")
total_miss = 0
for m in ["cn", "hk", "us"]:
    u = pd.read_csv(root / m / "universe.csv")
    have = sum(1 for s in u.symbol if (root / m / "bars" / f"{s}.parquet").exists())
    miss = len(u) - have
    total_miss += miss
    print(f"{m}:{have}/{len(u)} miss={miss}", flush=True)
print(f"TOTAL_MISS={total_miss}", flush=True)
PY
}

for attempt in $(seq 1 40); do
  echo "===== supervisor attempt $attempt $(date) =====" | tee -a "$LOG"
  coverage | tee -a "$LOG"
  miss=$("$PY" - <<'PY'
from pathlib import Path
import pandas as pd
root=Path("dataset")
miss=0
for m in ["cn","hk","us"]:
    u=pd.read_csv(root/m/"universe.csv")
    miss += sum(1 for s in u.symbol if not (root/m/"bars"/f"{s}.parquet").exists())
print(miss)
PY
)
  if [[ "$miss" -eq 0 ]]; then
    echo "DONE all markets covered" | tee -a "$LOG"
    exit 0
  fi
  # Lower concurrency a bit to reduce DNS / rate spikes
  "$PY" -u -B scripts/resume_download.py --markets cn hk us --workers 8 --rounds 3 >>"$LOG" 2>&1 || true
  echo "python exited; sleeping 3s" | tee -a "$LOG"
  sleep 3
done
echo "GAVE UP after 40 attempts" | tee -a "$LOG"
coverage | tee -a "$LOG"
exit 1
