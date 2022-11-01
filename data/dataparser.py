from abc import ABCMeta, abstractmethod
from pathlib import Path
from typing import List

import pandas as pd

from data.base import Interval, DataBlock
import tqdm
from concurrent.futures import ThreadPoolExecutor

from data.datasource import ParquetDataSource

FILE = Path(__file__).resolve()
PATH_DATA = FILE.parents[0]


class DataParser(metaclass=ABCMeta):
    """
    把不同外部数据源处理成统一的格式 data_block
    """

    @classmethod
    @abstractmethod
    def parse_data(cls, **kwargs) -> List[DataBlock]:
        """
        用于解析外部数据
        :param kwargs:
        :return:
        """
        pass


class ParquetDataParser(DataParser):
    """
    钱多多给的A股历史数据
    """

    @classmethod
    def parse_data(cls, f_name, **kwargs) -> List[DataBlock]:

        def worker(symbol):
            _df = df[df.symbol == symbol]
            _df.set_index("date", inplace=True)
            _df.sort_index(inplace=True)
            _df = _df[["adj_open", "adj_close", "adj_high", "adj_low", "volume", "amount", "trade_status"]]
            _df.columns = ["Open", "Close", "High", "Low", "Volume", "Amount", "trade_status"]
            _df.index.names = ["Date"]
            _data_block = DataBlock(data=_df, symbol=symbol, interval=Interval.DAILY)
            ParquetDataSource().save_data(_data_block)
            return _data_block

        df = pd.read_parquet(f_name, engine="pyarrow")  # TODO: 指定columns减少内存消耗
        symbol_cnts = df.symbol.value_counts()
        # 交易日>500天的symbol
        symbols = symbol_cnts[symbol_cnts > 1000].index.to_list()
        df.sort_values("symbol", inplace=True)
        # TODO 多线程并没有加快速度
        with ThreadPoolExecutor(max_workers=1) as executor:
            ret = list(tqdm.tqdm(executor.map(worker, symbols), total=len(symbols)))
        return ret


if __name__ == '__main__':
    ParquetDataParser.parse_data(f_name="/Users/yulin/workspace/extreme_quant/data/daily_200001_202209.parquet")
