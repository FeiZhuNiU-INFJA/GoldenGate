from enum import Enum

import pandas as pd
from pandas import DataFrame
from common.logger import LOGGER


class Interval(str, Enum):
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    MIN_1 = "min1"
    TICK = "tick"


class Exchange(str, Enum):
    """
    交易所
    """
    SSE = 'SSE'  # 上交所
    SZSE = 'SZSE'  # 深交所

    # 下面这些没有交易日历
    CFFEX = 'CFFEX'  # 中金所
    SHFE = 'SHFE'  # 上期所
    CZCE = 'CZCE'  # 郑商所
    DCE = 'DCE'  # 大商所
    INE = 'INE'  # 上能源


class Industry(str, Enum):
    """
    行业
    """
    pass


class Market(str, Enum):
    """
    市场类型（主板/创业板/科创板/CDR）
    根据业务规则，CDR可以在上交所主板或者深交所主板上市交易。
    CDR涨跌幅设定为10%，当交易所全天休市达到或超过7日的，其后首个交易日的涨跌幅比例为20%。
    在交易制度上，CDR除了竞价交易，还将引入做市商机制。
    """
    ZB = '主板'
    CYB = '创业板'
    KCB = '科创板'
    ZXB = '中小板'
    BJS = '北交所'
    # CDR = 'CDR'  # 太少了 先不要了


class IS_HS(str, Enum):
    # 是否沪深港通标的
    NO = 'N'  # 否
    HGT = 'H'  # 沪股通
    SGT = 'S'  # 深股通


class Asset:
    def __init__(self,
                 symbol: str,   # 600050
                 name: str,     # 联通
                 industry: Industry,
                 market: Market,
                 exchange: Exchange,
                 is_hs: IS_HS,
                 interval: Interval,
                 data: pd.DataFrame) -> None:
        super().__init__()
        self.symbol = symbol
        self.name = name
        self.industry = industry
        self.market = market
        self.exchange = exchange
        self.is_hs = is_hs
        self.interval = interval
        self.data = data


class DataBlock:
    """
    TODO 行情数据block 一个dataframe, 数据处理基于block来做
    """
    default_columns = ["Close", "Open", "High", "Low", "Volume"]

    def __init__(self, symbol: str,
                 data: DataFrame,
                 interval: Interval):
        self.symbol: str = symbol
        # TODO interval放在这合适吗
        self.interval: Interval = interval
        self.data: DataFrame = data

        # check data
        columns = self.data.columns.to_list()
        # TODO tick级别的数据 datetime长啥样？以及order book的数据长啥样
        for c in self.default_columns:
            if c not in columns:
                LOGGER.error(f"data should contain columns: {self.default_columns}")
                raise ValueError(f"data should contain columns: {self.default_columns}")


if __name__ == "__main__":
    print(Market.ZB == "主板")
