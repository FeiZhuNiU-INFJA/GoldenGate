from abc import ABCMeta
from pathlib import Path

"""
dataset
|-stock_cn
    |-daily
        |- ....
    |-symbol.csv
|-stock_nasdq


"""

DIR_DATA_DAILY = "daily"
DIR_DATA_HIST = "history"
FILE_SYMBOLS = "symbol.csv"
FILE_TRADE_CALENDAR = "calendar.csv"
FILE_INDEX = "index.csv"

DIR_PROJECT = Path(__file__).resolve().parents[1]
DIR_DATASET = DIR_PROJECT / 'dataset'


class AssetConfig(ABCMeta):
    HOME = DIR_DATASET / 'TBD'  # 根目录
    MARKET = "TBD"
    # DIR_HIST = "TBD"
    # F_SYMBOL = "TBD"
    # F_TRADE_CALENDAR = "TBD"


class SSEConfig(AssetConfig):
    HOME = DIR_DATASET / 'stock_sse'
    MARKET = "SSE"
    DIR_HIST = HOME / DIR_DATA_HIST  # 历史数据目录
    F_SYMBOL = HOME / FILE_SYMBOLS  # 所有股票代码文件
    F_TRADE_CALENDAR = HOME / FILE_TRADE_CALENDAR  # 交易日期文件
    F_INDEX = HOME / FILE_INDEX  # 指数列表，可以看到有哪些指数


class SZSEConfig(AssetConfig):
    HOME = DIR_DATASET / 'stock_szse'
    MARKET = "SZSE"
    DIR_HIST = HOME / DIR_DATA_HIST  # 历史数据目录
    F_SYMBOL = HOME / FILE_SYMBOLS  # 所有股票代码文件
    F_TRADE_CALENDAR = HOME / FILE_TRADE_CALENDAR  # 交易日期文件
    F_INDEX = HOME / FILE_INDEX  # 指数列表，可以看到有哪些指数


for config in [SSEConfig, SZSEConfig]:
    for x in config.__dict__:
        if x.startswith("DIR"):
            config.__dict__[x].mkdir(exist_ok=True, parents=True)
