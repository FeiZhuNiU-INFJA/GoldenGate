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

DIR_PROJECT = Path(__file__).resolve().parents[1]
DIR_DATASET = DIR_PROJECT / 'dataset'


class AssetConfig(ABCMeta):
    pass


class SSEConfig(AssetConfig):
    HOME = DIR_DATASET / 'stock_sse'
    DIR_HIST = HOME / DIR_DATA_HIST
    F_SYMBOL = HOME / FILE_SYMBOLS
    F_TRADE_CALENDAR = HOME / FILE_TRADE_CALENDAR


class SZSEConfig(AssetConfig):
    HOME = DIR_DATASET / 'stock_szse'
    DIR_HIST = HOME / DIR_DATA_HIST
    F_SYMBOL = HOME / FILE_SYMBOLS
    F_TRADE_CALENDAR = HOME / FILE_TRADE_CALENDAR


for config in [SSEConfig, SZSEConfig]:
    for x in config.__dict__:
        if x.startswith("DIR"):
            config.__dict__[x].mkdir(exist_ok=True, parents=True)
