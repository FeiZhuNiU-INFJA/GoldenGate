import glob
from pathlib import Path
import torch
from datetime import timedelta
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate import InitProcessGroupKwargs
import logging

init_kwargs = InitProcessGroupKwargs(timeout=timedelta(seconds=7200))
ACCELERATOR = Accelerator(kwargs_handlers=[init_kwargs])
DEVICE = ACCELERATOR.device
# DEVICE = 'cpu'
# 配置Logger
LOGGER = get_logger("my log")
LOGGER.setLevel("INFO")
formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
chlr = logging.FileHandler("log.txt")
chlr.setFormatter(formatter)
LOGGER.logger.addHandler(chlr)
LOGGER.info("init logger.", main_process_only=True)


# logging.basicConfig(filename='extreme.log', level=logging.INFO, format='%(asctime)s | %(name)s | %(levelname)s | %(message)s')
# LOGGER = logging.getLogger("extreme_quant")

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]

"""
dataset
|-stock_cn
    |-history_daily
        |- ....
    |-symbol.csv
    |-calendar.csv
    |- ....
|-stock_nasdq

"""

DIR_PROJECT = Path(__file__).resolve().parents[0]
DIR_DATASET = DIR_PROJECT / 'dataset'
STOCK_CN_HOME = DIR_DATASET / 'stock_cn'

# DIR_DATA_DAILY = "daily"
DIR_DATA_HIST_CN = STOCK_CN_HOME / "history_daily"
FILE_SYMBOLS_CN = STOCK_CN_HOME / "symbol.csv"
FILE_TRADE_CALENDAR_CN = STOCK_CN_HOME / "calendar.csv"
FILE_INDEX_CN = STOCK_CN_HOME / "index.csv"

DIR_DATA_HIST_CN.mkdir(exist_ok=True, parents=True)

BASE_FEATURES = ["open", "high", "low", "close", 'vol', 'amount', 'turnover_rate', 'volume_ratio']


