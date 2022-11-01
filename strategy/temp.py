from typing import List

import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch import optim
from torch.utils.data import WeightedRandomSampler
from torch.utils.data import Dataset, ConcatDataset, DataLoader
import torchvision
from pathlib import Path
import sys
FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  # ninja_pro

if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to
from strategy.model import MyLSTM

device = torch.device("cpu")
HEADER_TARGET = "Y"
SEQ_LENGTH = 32
IS_CLASSIFICATION = True
BATCH_SIZE = 128
NUM_FEATURES = 7  # TODO
HIDDEN_SIZE = 16
NUM_LAYERS = 2
DROPOUT = 0.05
DIRECTIONS = 1
N_CLASSES = 3
LEARNING_RATE = 0.0005
RESUME = False
EPOCHS = 10

params_dl = {'batch_size': BATCH_SIZE,
             'shuffle': False,  # TODO
             'drop_last': True,  # Disregard last incomplete batch
             'num_workers': 8}

params_dl_val = {'batch_size': 1,
                 'shuffle': False,
                 'drop_last': False,  # Disregard last incomplete batch
                 'num_workers': 8}


class CryptoDataset(Dataset):

    def __init__(self, data: pd.DataFrame,
                 head_features: List[str],
                 seq_length: int = SEQ_LENGTH,
                 symbol: str = "",
                 head_target: str = HEADER_TARGET,
                 is_classification: bool = IS_CLASSIFICATION,
                 one_hot_label: bool = False,
                 n_classes: int = N_CLASSES):
        self.symbol = symbol
        self.data = data
        self.target = head_target
        self.features = head_features
        self.seq_length = seq_length
        self.data_length = len(data)
        self.is_classification = is_classification
        self.one_hot_label = (self.is_classification and one_hot_label)  # for BCELoss
        self.n_classes = n_classes

    def __len__(self):
        return self.data_length - self.seq_length

    def __getitem__(self, idx):
        x = self.data[idx:idx + self.seq_length][self.features].values
        y = self.data.iloc[idx + self.seq_length - 1][self.target]
        if self.is_classification and self.one_hot_label:
            y = F.one_hot(torch.tensor(y), num_classes=self.n_classes)
        return x, y


def get_xy(f_name) -> pd.DataFrame:
    df = pd.read_parquet(f_name, engine="pyarrow")

    df = df[["open", "close", "high", "low", "volume", "number_of_trades", "taker_buy_volume"]]
    df.sort_index(inplace=True)
    print(f"before fullfill: {len(df)}")
    # 补全漏掉的时间
    new_date_range = pd.date_range(start=df.index[0], end=df.index[-1], freq="min")
    print(f"after fullfill: {len(new_date_range)}")
    # 漏掉的数据用上一个时间点的数据来填充
    df.reindex(new_date_range, method="ffill")
    df["return"] = df["close"].shift(-1) / df["close"] - 1
    df[HEADER_TARGET] = df["return"].apply(lambda x: 1 if x > 0.002 else (2 if x < -0.002 else 0))  # 1 涨 2 跌
    df = df.drop(columns=["return"])
    df["open"] = np.log1p(df["open"])
    df["close"] = np.log1p(df["close"])
    df["high"] = np.log1p(df["high"])
    df["low"] = np.log1p(df["low"])
    df["volume"] = np.log1p(df["volume"])
    df["number_of_trades"] = np.log1p(df["number_of_trades"])
    df["taker_buy_volume"] = np.log1p(df["taker_buy_volume"])
    # TODO  add indicators
    print(df.head(15))
    print(df[HEADER_TARGET].value_counts())
    return df


if __name__ == '__main__':

    model = MyLSTM(
        input_size=NUM_FEATURES,
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        dropout_prob=DROPOUT,
        directions=DIRECTIONS,
        is_classification=IS_CLASSIFICATION,
        n_classes=N_CLASSES,
    ).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
    if IS_CLASSIFICATION:
        # criterion = torchvision.ops.focal_loss.sigmoid_focal_loss
        criterion = nn.CrossEntropyLoss(weight=torch.tensor([1, 1., 1.]).float()).to(device)
    else:
        criterion = nn.MSELoss().to(device)

    f_btc = "/Users/yulin/workspace/extreme_quant/data/raw_data/btc.parquet"
    f_eth = "/Users/yulin/workspace/extreme_quant/data/raw_data/eth.parquet"
    training_datasets = []
    validation_datasets = []
    cnts = 0
    for f in [f_btc, f_eth]:
        df = get_xy(f)
        cnts+=len(df)
        cols = df.columns.tolist()
        cols.remove(HEADER_TARGET)
        # df = df.iloc[0:10000]
        print(cols)
        training_dataset = CryptoDataset(data=df.iloc[0:int(len(df) * 0.96)], head_features=cols)
        training_datasets.append(training_dataset)

        validation_dataset = CryptoDataset(data=df.iloc[int(len(df) * 0.96):], head_features=cols)
        validation_datasets.append(validation_dataset)

    sampler = WeightedRandomSampler(weights=[1,25,25], num_samples=cnts, replacement=True)
    training_dl = DataLoader(ConcatDataset(training_datasets),  sampler=sampler, **params_dl)
    validation_dl = DataLoader(ConcatDataset(validation_datasets), **params_dl_val)

    model.train_model(training_dl=training_dl, validation_dl=validation_dl, optimizer=optimizer, criterion=criterion,
                      epochs=EPOCHS, batch_size=params_dl["batch_size"], validate_batch_size=params_dl_val["batch_size"],
                      is_classification=IS_CLASSIFICATION, use_bceloss=False, validate_every=1)
