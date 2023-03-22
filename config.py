from data.annotation import BuySellPointAnnotation
import glob
from pathlib import Path
import torch

from strategy.model import MyLSTM

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

DIR_PROJECT = Path(__file__).resolve().parents[1]
DIR_DATASET = DIR_PROJECT / 'dataset'
STOCK_CN_HOME = DIR_DATASET / 'stock_cn'

# DIR_DATA_DAILY = "daily"
DIR_DATA_HIST_CN = STOCK_CN_HOME / "history_daily"
FILE_SYMBOLS_CN = STOCK_CN_HOME / "symbol.csv"
FILE_TRADE_CALENDAR_CN = STOCK_CN_HOME / "calendar.csv"
FILE_INDEX_CN = STOCK_CN_HOME / "index.csv"

DIR_DATA_HIST_CN.mkdir(exist_ok=True, parents=True)

BASE_FEATURES = ["open", "high", "low", "close", 'vol', 'amount']

anno1 = BuySellPointAnnotation(quote_change=0.2, soft_percent=0.02, soft_eta=0.9, min_gap=5)
anno2 = BuySellPointAnnotation(quote_change=0.3, soft_percent=0.03, soft_eta=0.9, min_gap=5)

# 1st model
SEQ_LENGTH = 64
FEATURES_HEAD = BASE_FEATURES
HIDDEN_SIZE = 16
NUM_LAYERS = 2
DROPOUT = 0.1
DIRECTIONS = 2
IS_CLASSIFICATION = True
device = torch.device("cpu")
model1 = MyLSTM(
    input_size=len(FEATURES_HEAD),
    hidden_size=HIDDEN_SIZE,
    num_layers=NUM_LAYERS,
    dropout_prob=DROPOUT,
    directions=DIRECTIONS,
    is_classification=IS_CLASSIFICATION,
    n_classes=anno1.n_class(),
    use_bceloss=False,
    device=device,
    seq_length=SEQ_LENGTH,
    weight="mylstm_64.pt",
    # weight="mylstm_last.pt",
).to(device)
model1.eval()


model2 = MyLSTM(
    input_size=len(FEATURES_HEAD),
    hidden_size=HIDDEN_SIZE,
    num_layers=NUM_LAYERS,
    dropout_prob=DROPOUT,
    directions=DIRECTIONS,
    is_classification=IS_CLASSIFICATION,
    n_classes=anno1.n_class(),
    use_bceloss=False,
    device=device,
    seq_length=128,
    weight="mylstm_128.pt",
    # weight="mylstm_last.pt",
).to(device)
model2.eval()
