from typing import List

from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from data.annotation import Annotation

class SingleSymbolDataset(Dataset):

    def __init__(self,
                 f_hist: str,
                 anno: Annotation,
                 features_head: List[str],
                 seq_len,
                 start_date,
                 end_date,
                 with_aug=False,
                 ):
        self.f_hist = f_hist
        self.anno = anno  # 打标签的方法
        self.features_head = features_head  # 特征对应的head
        self.seq_len = seq_len
        self.with_aug = with_aug

        self.hist = pd.read_csv(f_hist)
        # features_head有缺失
        if not all(x in list(self.hist.columns) for x in self.features_head):
            self.hist = self.hist[0:0]
            return
        # 根据start_date, end_date过滤出数据
        self.hist.set_index("trade_date", inplace=True)
        self.hist.sort_index(inplace=True)
        self.hist.index = pd.to_datetime(self.hist.index, format='%Y%m%d')
        self.hist = self.hist[start_date:end_date]

    def __len__(self):
        return max(len(self.hist) - self.seq_len, 0)
    
    @classmethod
    def preprocess(cls, x):
        x[:, 0:4] /= x[0][0]  # OHCL
        x[:, 4] /= x[0][4]    # vol  
        x[:, 5] /= x[0][5]    # amount  
        return x

    def __getitem__(self, idx):
        x = self.hist[idx:idx + self.seq_len]
        x = x[self.features_head].values
        if self.with_aug:
            x = x + x * (np.random.random(x.shape) / 500 - 0.001)  # 添加0.1%的噪声
        # 归一化 以第一天的开盘价为基准
        x = self.preprocess(x)
        # x[:, 0:6] -= 1
        y = self.hist.iloc[idx + self.seq_len - 1][self.anno.head_label]
        y = self.anno.label_to_class(y)
        return x, y


if __name__ == "__main__":
    pass
