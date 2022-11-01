from enum import Enum

from pandas import DataFrame

from common.logger import LOGGER


class Interval(Enum):
    DAILY = 0
    WEEKLY = 1
    MONTHLY = 2
    MIN_1 = 3
    TICK = 4


class DataBlock:
    """
    行情数据block 一个dataframe，数据处理基于block来做
    """
    default_columns = ["Close", "Open", "High", "Low", "Volume"]

    def __init__(self, symbol: str,
                 data: DataFrame,
                 interval: Interval):
        self.symbol: str = symbol
        self.interval: Interval = interval
        self.data: DataFrame = data

        # check data
        columns = self.data.columns.to_list()
        # TODO tick级别的数据 datetime长啥样？以及order book的数据长啥样
        for c in self.default_columns:
            if c not in columns:
                LOGGER.error(f"data should contain columns: {self.default_columns}")
                raise ValueError(f"data should contain columns: {self.default_columns}")
