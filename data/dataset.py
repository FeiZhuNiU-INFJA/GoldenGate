from typing import List

import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import pandas as pd
import tqdm
import glob
import torch.nn.functional as F


class SingleSymbolDataset(Dataset):

    def __init__(self,
                 f_hist: str,
                 label_head: str,
                 features_head: List[str],
                 start_date,
                 end_date,
                 seq_len,
                 is_classification: bool = True,
                 is_one_hot_label: bool = False,
                 func_label_to_class=None,  # 用于把label转换成类别的方法
                 n_classes: int = 3,
                 ):
        self.f_hist = f_hist
        self.label_head = label_head  # 标签对应的head
        self.features_head = features_head  # 特征对应的head
        # self.start_date = start_date      # 20000101 这样的格式
        # self.end_date = end_date
        self.seq_len = seq_len
        self.func_label_to_class = func_label_to_class
        self.is_classification = is_classification and func_label_to_class is not None
        self.is_one_hot_label = is_one_hot_label
        self.n_classes = n_classes

        self.hist = pd.read_csv(f_hist)
        # 根据start_date, end_date过滤出数据
        self.hist.set_index("trade_date", inplace=True)
        self.hist.sort_index(inplace=True)
        self.hist.index = pd.to_datetime(self.hist.index, format='%Y%m%d')
        self.hist = self.hist[start_date:end_date]

    def __len__(self):
        return max(len(self.hist) - self.seq_len, 0)

    def __getitem__(self, idx):
        x = self.hist[idx:idx + self.seq_len][self.features_head].values
        y = self.hist.iloc[idx + self.seq_len - 1][self.label_head]

        if self.is_classification:
            y = self.func_label_to_class(y)
            if self.is_one_hot_label:
                y = F.one_hot(torch.tensor(y), num_classes=self.n_classes)

        return x, y


if __name__ == "__main__":
    # 时间序列长度
    SEQ_LEN = 64
    # 训练、验证数据的时间跨度
    TRAIN_START_DATE = "20000101"
    TRAIN_END_DATE = "20211231"
    VAL_START_DATE = "20220101"
    VAL_END_DATE = "20220630"

    train_dl_params = {
        'batch_size': 16,
        'shuffle': True,
        'drop_last': True,  # Disregard last incomplete batch
        'num_workers': 8
    }
    val_dl_params = {
        'batch_size': 1,
        'shuffle': False,
        'drop_last': False,  # Disregard last incomplete batch
        'num_workers': 8
    }

    training_datasets = []
    validation_datasets = []

    for f_sse in tqdm.tqdm(glob.glob("/Users/yulin/workspace/extreme_quant/dataset/stock_sse/history/*.csv")):
        training_datasets.append(
            SingleSymbolDataset(f_hist=f_sse,
                                label_head="BuySellPoint_qc_0.2_sp_0.02",
                                features_head=["open", "high", "low", "close"],
                                start_date=TRAIN_START_DATE,
                                end_date=TRAIN_END_DATE,
                                seq_len=SEQ_LEN)
        )
        validation_datasets.append(
            SingleSymbolDataset(f_hist=f_sse,
                                label_head="BuySellPoint_qc_0.2_sp_0.02",
                                features_head=["open", "high", "low", "close"],
                                start_date=VAL_START_DATE,
                                end_date=VAL_END_DATE,
                                seq_len=SEQ_LEN)
        )

    for f_szse in tqdm.tqdm(glob.glob("/Users/yulin/workspace/extreme_quant/dataset/stock_szse/history/*.csv")):
        training_datasets.append(
            SingleSymbolDataset(f_hist=f_szse,
                                label_head="BuySellPoint_qc_0.2_sp_0.02",
                                features_head=["open", "high", "low", "close"],
                                start_date=TRAIN_START_DATE,
                                end_date=TRAIN_END_DATE,
                                seq_len=SEQ_LEN)
        )
        validation_datasets.append(
            SingleSymbolDataset(f_hist=f_szse,
                                label_head="BuySellPoint_qc_0.2_sp_0.02",
                                features_head=["open", "high", "low", "close"],
                                start_date=VAL_START_DATE,
                                end_date=VAL_END_DATE,
                                seq_len=SEQ_LEN)
        )

    training_dl = DataLoader(ConcatDataset(training_datasets), **train_dl_params)
    validation_dl = DataLoader(ConcatDataset(validation_datasets), **val_dl_params)
