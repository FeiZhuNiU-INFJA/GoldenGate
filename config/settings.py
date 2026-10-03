"""Central configuration for the multi-market quant pipeline."""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DIR_DATASET = ROOT / "dataset"
DIR_CHECKPOINTS = ROOT / "checkpoints"
DIR_REPORTS = ROOT / "reports"

DIR_DATASET.mkdir(exist_ok=True, parents=True)
DIR_CHECKPOINTS.mkdir(exist_ok=True, parents=True)
DIR_REPORTS.mkdir(exist_ok=True, parents=True)

MARKETS = ("cn", "hk", "us", "ndx")
MARKET_TO_ID = {"cn": 0, "hk": 1, "us": 2, "ndx": 3}
ID_TO_MARKET = {v: k for k, v in MARKET_TO_ID.items()}

# Unified feature columns available across markets (Akshare).
FEATURE_COLS = ["open", "high", "low", "close", "volume", "amount"]
N_FEATURES = len(FEATURE_COLS)

# Label horizons and excess-return thresholds vs market benchmark.
HORIZONS = (5, 20)
LABEL_THRESHOLDS = {
    5: 0.015,   # ±1.5%
    20: 0.04,   # ±4%
}
LABEL_COLS = [f"y_{h}d" for h in HORIZONS]
EXRET_COLS = [f"exret_{h}d" for h in HORIZONS]
RET_COLS = [f"ret_{h}d" for h in HORIZONS]

# Class ids: 0=neutral, 1=buy, 2=sell
N_CLASSES = 3
CLASS_NAMES = ("neutral", "buy", "sell")

# Benchmark index codes used when downloading (market-specific adapters map these).
BENCHMARKS = {
    "cn": "000300",  # CSI 300
    "hk": "HSI",     # Hang Seng
    "us": ".INX",    # S&P 500 (Akshare/Eastmoney style; may be remapped in client)
    "ndx": ".NDX",   # Nasdaq-100
}

# Sequence / model defaults
SEQ_LEN = 128
EMBED_DIM = 64
N_LAYERS = 4
N_HEADS = 4
DROPOUT = 0.1

# Train / val / test by label date t
TRAIN_START = "2015-01-01"
TRAIN_END = "2022-12-31"
VAL_START = "2023-01-01"
VAL_END = "2023-12-31"
TEST_START = "2024-01-01"
TEST_END = "2099-12-31"

# Inference score blend
SCORE_WEIGHT_5D = 0.4
SCORE_WEIGHT_20D = 0.6
LOSS_WEIGHT_5D = 0.5
LOSS_WEIGHT_20D = 0.5

# Download throttling / concurrency
DOWNLOAD_SLEEP_SEC = 0.35
DOWNLOAD_MAX_RETRIES = 3
DOWNLOAD_START_DATE = "20140101"
DOWNLOAD_WORKERS = 8  # parallel bar fetchers (CN=processes, HK/US=threads)

# Training defaults
BATCH_SIZE = 256
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 1e-4
EPOCHS = 20
NUM_WORKERS = 4

# Backtest defaults
BACKTEST_TOP_N = 20
BACKTEST_REBALANCE = "W-FRI"  # weekly Friday
