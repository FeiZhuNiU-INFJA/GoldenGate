from abc import ABCMeta, abstractmethod
from enum import Enum
import numpy as np
import pandas as pd
from pathlib import Path
import sys
import plotly.graph_objects as go

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]

if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to
from common.logger import LOGGER

HEAD_LABEL = "Target"


class Annotation(metaclass=ABCMeta):
    """
    用于生成label的基类
    """

    # def __init__(self, data: pd.DataFrame) -> None:
    #     self.data = data
    #     self.data_with_label = None

    @classmethod
    @abstractmethod
    def _labeling(cls, data: pd.DataFrame, **kwargs) -> pd.Series:
        pass

    @classmethod
    def generate_labeled_data(cls, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        调用子类的_labeling方法
        :param data:
        :param kwargs:
        :return:
        """
        ret = data.copy()
        LOGGER.info(f"Labeling using {cls.__name__} ")
        label = cls._labeling(ret, **kwargs)
        if label is None or len(label) != len(data):
            LOGGER.error(f"{cls.__name__} Labeling not working correctly")
        else:
            label.name = HEAD_LABEL
            ret[HEAD_LABEL] = label.values
        LOGGER.info(f"Labeling complete")
        return ret

    @classmethod
    @abstractmethod
    def visualize(cls, data_with_label: pd.DataFrame, **kwargs):
        """
        把self.data_with_label可视化出来
        """
        pass


class BuySellPointAnnotation(Annotation):

    @classmethod
    def _labeling(cls, data: pd.DataFrame,
                  quote_change=0.2,
                  soft_percent=0.,
                  soft_eta=1.,
                  min_gap=1):
        """
        quote_change: 涨跌幅百分比
        min_interval: 买卖最小间隔
        soft_percent: 买卖点附近差值在soft_percent以内的也算做买卖点，可以付权重
        soft_eta: soft对应的衰减率

        1 表示买  -1表示卖
        """
        min_gap = max(1, min_gap)
        soft_percent = min(soft_percent, min_gap)

        closes = list(np.squeeze((data["Close"]).values))
        print(len(closes))
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
                    _labels[j] = (0 if undo else (round(_labels[j + 1] * soft_eta, 2)))
                else:
                    break
            # 往后看
            for j in range(_idx + 1, _idx_cur - 1, 1):
                if abs(closes[j] - closes[_idx]) / closes[_idx] <= soft_percent:
                    _labels[j] = (0 if undo else round(_labels[j - 1] * soft_eta, 2))
                else:
                    break

        for cur_idx in range(1, len(closes)):
            close = closes[cur_idx]
            if close >= high:
                high = close
                idx_high = cur_idx
                if high / low >= (
                        1 + quote_change) and cur_idx - idx_low >= min_gap and idx_low - idx_last_point >= min_gap:
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
                        high - low) / high >= quote_change and cur_idx - idx_high >= min_gap and idx_high - idx_last_point >= min_gap:
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
                            high - close) / high >= quote_change and cur_idx - idx_high >= min_gap and idx_high - idx_last_point >= min_gap:
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
                            1 + quote_change) and cur_idx - idx_low >= min_gap and idx_low - idx_last_point >= min_gap:
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

    @classmethod
    def visualize(cls, data_with_label: pd.DataFrame, **kwargs):
        hist = data_with_label
        # 把走势画出来
        fig = go.Figure(
            data=go.Scatter(x=hist.index, y=hist['Close'], mode='lines', name=f'Price_{kwargs.get("symbol", "")}'))
        # 把target不为零的点画出来
        hist2 = hist[hist["Target"] != 0]
        fig.add_scatter(x=hist2.index, y=hist2['Close'], mode='markers', marker_color=hist2['Target'], name="Label")
        fig.show()


class ReturnAnnotation(Annotation):
    """
    第二天的return
    """

    @classmethod
    def _labeling(cls, data: pd.DataFrame, **kwargs) -> pd.Series:
        return data["Close"].shift(-1) / data["Close"] - 1

    @classmethod
    def visualize(cls, data_with_label: pd.DataFrame, **kwargs):
        pass


if __name__ == '__main__':
    pass
