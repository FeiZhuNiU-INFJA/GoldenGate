from data.annotation import BuySellPointAnnotation
import glob
from pathlib import Path

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]

BASE_FEATURES = ["open", "high", "low", "close"]

anno1 = BuySellPointAnnotation(quote_change=0.2, soft_percent=0.02, soft_eta=0.9, min_gap=5)

CSV_SSE_STOCKS = list(glob.glob(f"{ROOT}/dataset/stock_sse/history/*.csv"))
CSV_SZSE_STOCKS = list(glob.glob(f"{ROOT}/dataset/stock_szse/history/*.csv"))

