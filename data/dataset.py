from typing import List

import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import pandas as pd
from torch.utils.data.dataset import T_co


class SingleSymbolDataset(Dataset):

    def __init__(self,
                 f_hist: str,
                 df_calendar: pd.DataFrame,
                 label_head: str,
                 start_date,
                 end_date,
                 seq_len):
        self.f_hist = f_hist
        self.df_calendar = df_calendar
        self.label_head = label_head
        self.start_date = start_date
        self.end_date = end_date
        self.seq_len = seq_len

        df_hist = pd.read_csv(f_hist)
        dh_hist.set_index

    def __len__(self):


    def __getitem__(self, index):
        pass


class MyDataset(Dataset):

    def __init__(self, data: pd.DataFrame,
                 seq_length, head_features,
                 symbol,
                 head_target="Target",
                 is_classification=False,
                 one_hot_label=False,
                 n_classes=3):
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
        if self.is_classification:
            if y > 0.1:
                y = 2  # sell
            elif y < -0.1:
                y = 1  # buy
            else:
                y = 0  # do nothing
            if self.one_hot_label:
                y = F.one_hot(torch.tensor(y), num_classes=self.n_classes)
        return x, y
