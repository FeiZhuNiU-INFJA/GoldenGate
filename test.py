from typing import Tuple
import numpy as np
import torch
import pandas as pd

# import config
from hubs import anno1, model1, model2
from strategy.model import MyLSTM
from config import BASE_FEATURES, DIR_DATA_HIST_CN
# SEQ_LENGTH = 64
# HIDDEN_SIZE = 16
# NUM_LAYERS = 2
# DROPOUT = 0.1
# DIRECTIONS = 2
# IS_CLASSIFICATION = True
# device = torch.device("cpu")


if __name__ == '__main__':
    # f_stock = "600000.SH.csv"
    f_stock = f"{DIR_DATA_HIST_CN}/002230.SZ.csv"
    # f_stock = f"{DIR_DATA_HIST_CN}/003028.SZ.csv"
    # f_stock = f"{DIR_DATA_HIST_CN}/300573.SZ.csv"
    # f_stock = f"{DIR_DATA_HIST_CN}/600818.SH.csv"
    # f_stock = f"{DIR_DATA_HIST_CN}/688331.SH.csv"
    # f_stock = f"{DIR_DATA_HIST_CN}/601360.SH.csv"
    df = pd.read_csv(f_stock)
    df.set_index("trade_date", inplace=True)
    df.index = pd.to_datetime(df.index, format='%Y%m%d')
    df = df["20220601":]
    df = df[BASE_FEATURES + [anno1.head_label]]  # trade_date 用于可视化

    model = model2
    result = model.test_model(
        data=df,
        threshold=None,
        head_label=anno1.head_label,
        head_features=BASE_FEATURES
    )
    result.to_csv("test.csv")
    anno1.visualize(result)
