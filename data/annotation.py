import traceback
from abc import ABCMeta, abstractmethod
from enum import Enum
from typing import Optional
import numpy as np
import pandas as pd
from pathlib import Path
import sys
import plotly.graph_objects as go
from p_tqdm import p_umap
import glob
FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]

if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to
from common.logger import LOGGER


class Classiable(metaclass=ABCMeta):
    @classmethod
    @abstractmethod
    def label_to_class(cls, label):
        """
        如果是分类任务，映射到哪一类
        :param label:
        :return:
        """
        pass

    @classmethod
    @abstractmethod
    def n_class(cls):
        """
        如果是分类任务，可以分成几类
        :return:
        """
        pass


class Annotation(metaclass=ABCMeta):
    """
    用于打标签的基类
    """
    @property
    @abstractmethod
    def head_label(self):
        """
        表头
        """
        return "should be reset in subclass"

    @abstractmethod
    def _labeling(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        pass

    # TODO 参数放到构造函数里面去
    def generate_data_with_label(self, data: pd.DataFrame, **kwargs) -> Optional[pd.DataFrame]:
        data = data.copy()
        LOGGER.info(f"Labeling using {self.__class__.__name__} ")
        label = self._labeling(data, **kwargs)
        if label is None or len(label) != len(data):
            LOGGER.error(f"{self.__class__.__name__} Labeling not working correctly")
            return None
        else:
            label.name = self.head_label
            data[self.head_label] = label.values
        return data

    @abstractmethod
    def visualize(self, data_with_label: pd.DataFrame, **kwargs):
        """
        把self.data_with_label可视化出来
        """
        pass


class BuySellPointAnnotation(Annotation, Classiable):

    def __init__(self,
                 quote_change=0.2,
                 soft_percent=0.,
                 soft_eta=1.,
                 min_gap=1):
        super().__init__()
        """
        quote_change: 涨跌幅百分比
        min_interval: 买卖最小间隔
        soft_percent: 买卖点附近差值在soft_percent以内的也算做买卖点，可以付权重
        soft_eta: soft对应的衰减率
        """
        self.quote_change = quote_change
        self.soft_percent = soft_percent
        self.soft_eta = soft_eta
        self.min_gap = min_gap

    @property
    def head_label(self):
        return f"BuySellPoint_qc_{self.quote_change}_sp_{self.soft_percent}"

    def _labeling(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """
        1 表示买  -1表示卖
        """
        min_gap = max(1, self.min_gap)
        soft_percent = min(self.soft_percent, min_gap)

        closes = list(np.squeeze((data["close"]).values))
        # print(len(closes))
        labels = [0] * len(closes)
        LABEL_SELL = -1
        LABEL_BUY = 1
        high = closes[0]
        low = closes[0]
        idx_high = 0
        idx_low = 0

        class Status(Enum):
            Nan = 0
            Bought = 1
            Sold = 2

        status = Status.Nan  # 记录最近一次处理过的是买点还是卖点

        idx_last_point = -100  # 记录最近一次处理的点的index

        def process_soft(_idx, _idx_cur, _idx_pre, _labels, undo=False):
            """
            _idx: 买卖点下标
            _idx_cur: 当前下标，往后看的时候，不会超过这个下标
            _idx_pre": 上一个买卖点，往前看的时候，不会超过这个
            处理买卖点附近的label，
            与买卖点相差不到soft_percent的点，都给label，以soft_eta衰减
            """
            # 往前看
            for j in range(_idx - 1, max(0, _idx_pre + min_gap), -1):
                if abs(closes[j] - closes[_idx]) / closes[_idx] <= soft_percent:
                    _labels[j] = (0 if undo else (round(_labels[j + 1] * self.soft_eta, 2)))
                else:
                    break
            # 往后看
            for j in range(_idx + 1, _idx_cur - 1, 1):
                if abs(closes[j] - closes[_idx]) / closes[_idx] <= soft_percent:
                    _labels[j] = (0 if undo else round(_labels[j - 1] * self.soft_eta, 2))
                else:
                    break

        for cur_idx in range(1, len(closes)):
            close = closes[cur_idx]
            if close >= high:
                high = close
                idx_high = cur_idx
                if high / low >= (
                        1 + self.quote_change) and cur_idx - idx_low >= min_gap and idx_low - idx_last_point >= min_gap:
                    labels[idx_low] = LABEL_BUY  # 买点
                    process_soft(idx_low, cur_idx, idx_last_point, labels)
                    if status == Status.Bought:
                        labels[idx_last_point] = 0
                        process_soft(idx_last_point, idx_low, 0, labels, undo=True)
                    idx_last_point = idx_low
                    status = Status.Bought

            elif close <= low:
                low = close
                idx_low = cur_idx
                if (
                        high - low) / high >= self.quote_change and cur_idx - idx_high >= min_gap and idx_high - idx_last_point >= min_gap:
                    labels[idx_high] = LABEL_SELL  # 卖点
                    process_soft(idx_high, cur_idx, idx_last_point, labels)
                    if status == Status.Sold:
                        labels[idx_last_point] = 0
                        process_soft(idx_last_point, idx_high, 0, labels, undo=True)
                    idx_last_point = idx_high
                    status = Status.Sold

            else:
                if status == Status.Bought:
                    if (
                            high - close) / high >= self.quote_change and cur_idx - idx_high >= min_gap and idx_high - idx_last_point >= min_gap:
                        labels[idx_high] = LABEL_SELL  # 卖点
                        process_soft(idx_high, cur_idx, idx_last_point, labels)
                        if status == Status.Sold:
                            labels[idx_last_point] = 0
                            process_soft(idx_last_point, idx_high, 0, labels, undo=True)
                        idx_last_point = idx_high
                        status = Status.Sold
                        low = close
                        idx_low = cur_idx

                elif status == Status.Sold:
                    if close / low >= (
                            1 + self.quote_change) and cur_idx - idx_low >= min_gap and idx_low - idx_last_point >= min_gap:
                        labels[idx_low] = LABEL_BUY  # 买点
                        process_soft(idx_low, cur_idx, idx_last_point, labels)
                        if status == Status.Bought:
                            labels[idx_last_point] = 0
                            process_soft(idx_last_point, idx_low, 0, labels, undo=True)
                        idx_last_point = idx_low
                        status = Status.Bought
                        high = close
                        idx_high = cur_idx
        return pd.Series(labels)

    def visualize(self, data_with_label: pd.DataFrame, **kwargs):
        hist = data_with_label
        # 把走势画出来
        fig = go.Figure(
            data=go.Scatter(x=hist.index, y=hist['close'], mode='lines', name=f'Price_{kwargs.get("symbol", "")}'))
        # 把target不为零的点画出来
        hist2 = hist[hist[self.head_label] != 0]
        fig.add_scatter(x=hist2.index, y=hist2['close'], mode='markers', marker_color=hist2[self.head_label], name="Label")
        fig.show()

    @classmethod
    def label_to_class(cls, label):
        if label == 0:
            return 0
        if label > 0:
            return 1
        else:
            return 2

    @classmethod
    def n_class(cls):
        """
        0, 1 买, 2 卖
        :return:
        """
        return 3


# class ReturnAnnotation(Annotation):
#     """
#     第二天的return
#     """

#     @classmethod
#     def _labeling(cls, data: pd.DataFrame, **kwargs) -> pd.Series:
#         return data["close"].shift(-1) / data["close"] - 1

#     @classmethod
#     def visualize(cls, data_with_label: pd.DataFrame, **kwargs):
#         pass

def worker_anno_buysellpoint(f_csv):
    try:
        df = pd.read_csv(f_csv)
        anno = BuySellPointAnnotation(quote_change=0.2, soft_percent=0.02, soft_eta=0.9, min_gap=5)
        df = anno.generate_data_with_label(df)
        if df is not None:
            df.to_csv(f_csv, index=False)
    except:
        print(f_csv, traceback.format_exc())


if __name__ == '__main__':

    SSE_stocks = list(glob.glob(f"{ROOT}/dataset/stock_sse/history/*.csv"))
    SZSE_stocks = list(glob.glob(f"{ROOT}/dataset/stock_szse/history/*.csv"))

    p_umap(worker_anno_buysellpoint, SSE_stocks + SZSE_stocks, desc="Label Buy Sell Point", num_cpus=8)
    # anno.generate_labeled_data(df, f_target="600000.SH.csv")
