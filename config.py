from data.annotation import BuySellPointAnnotation
import glob
from pathlib import Path
import torch

from strategy.model import MyLSTM

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]

BASE_FEATURES = ["open", "high", "low", "close", 'vol', 'amount']

anno1 = BuySellPointAnnotation(quote_change=0.2, soft_percent=0.02, soft_eta=0.9, min_gap=5)

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
    weight="mylstm_epoch_12.pt",
    # weight="mylstm_last.pt",
).to(device)
