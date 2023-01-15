from typing import Tuple
import numpy as np
import torch
import pandas as pd

from config import anno1
from strategy.model import MyLSTM

SEQ_LENGTH = 64
FEATURES_HEAD = ["open", "high", "low", "close"]
HIDDEN_SIZE = 16
NUM_LAYERS = 2
DROPOUT = 0.1
DIRECTIONS = 2
IS_CLASSIFICATION = True
device = torch.device("cpu")


if __name__ == '__main__':
    # f_stock = "600000.SH.csv"
    f_stock = "dataset/stock_sse/history/601375.SH.csv"
    df = pd.read_csv(f_stock)
    df.set_index("trade_date", inplace=True)
    df.index = pd.to_datetime(df.index, format='%Y%m%d')
    df = df[FEATURES_HEAD + [anno1.head_label]]  # trade_date 用于可视化

    model = MyLSTM(
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
        weight="LSTM_seql64_best_baseline.pt",
    ).to(device)

    result = model.test_model(
        data=df,
        threshold=0.8,
        head_label=anno1.head_label,
        head_features=FEATURES_HEAD
    )

    anno1.visualize(result)
