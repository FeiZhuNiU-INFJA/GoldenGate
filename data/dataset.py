from typing import List

import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import pandas as pd
import tqdm
import glob
import torch.nn.functional as F

from data.config import DIR_DATA_HIST_CN


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
        # TODO 归一化
        x[:, 0:4] /= x[0][0]
        x[:, 4] /= x[0][4]
        x[:, 5] /= x[0][5]
        y = self.hist.iloc[idx + self.seq_len - 1][self.label_head]

        if self.is_classification:
            y = self.func_label_to_class(y)
            if self.is_one_hot_label:
                y = F.one_hot(torch.tensor(y), num_classes=self.n_classes)

        return x, y


if __name__ == "__main__":
    pass
