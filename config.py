import glob
from pathlib import Path
import torch
import logging


logging.basicConfig(filename='extreme.log', level=logging.INFO, format='%(asctime)s | %(name)s | %(levelname)s | %(message)s')
LOGGER = logging.getLogger("extreme_quant")

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

BASE_FEATURES = ["open", "high", "low", "close", 'vol', 'amount']


# 1st model
# SEQ_LENGTH = 64
# FEATURES_HEAD = BASE_FEATURES
# FEATURES_HEAD2 = BASE_FEATURES + ["turnover_rate", "volume_ratio"]
# HIDDEN_SIZE = 16
# NUM_LAYERS = 2
# DROPOUT = 0.1
# DIRECTIONS = 2
# IS_CLASSIFICATION = True
DEVICE = torch.device("cpu")


# model3 = MyLSTM(
#     input_size=8,
#     hidden_size=16,
#     num_layers=2,
#     dropout_prob=0.1,
#     directions=2,
#     is_classification=True,
#     n_classes=3,
#     use_bceloss=False,
#     device=device,
#     seq_length=128,
#     weight="mylstm_128_HEAD2.pt",
# ).to(device)
# model3.eval()
