from abc import ABCMeta, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd

from common.logger import LOGGER
from data.base import Interval, DataBlock

FILE = Path(__file__).resolve()
PATH_DATA = FILE.parents[0]


class DataSource(metaclass=ABCMeta):

    @abstractmethod
    def get_data_block(self, symbol, interval=Interval.DAILY,
                       start_date=None, end_date=None, **kwargs) -> Optional[DataBlock]:
        """
        实现读取数据块的功能
        :param symbol:
        :param interval:
        :param start_date:
        :param end_date:
        :param kwargs:
        :return:
        """
        pass

    @abstractmethod
    def save_data(self, data_block: DataBlock, **kwargs):
        """
        实现存数据的功能
        :param data_block:
        :param kwargs:
        :return:
        """
        pass

    @abstractmethod
    def update_data(self, data_block: DataBlock, **kwargs):
        """
        实现更新数据的功能
        :param data_block:
        :param kwargs:
        :return:
        """
        pass


class ParquetDataSource(DataSource):
    """
    用于把数据都存到本地的parquet文件
    一只票一个parquet文件
    """

    _F_PARQUET_PATTERN = str(PATH_DATA / "parquet" / "{}_{}.parquet")  # {symbol}_{interval}

    def __init__(self) -> None:
        super().__init__()
        target_folder = PATH_DATA / "parquets"
        if not target_folder.exists():
            target_folder.mkdir()

    def get_data_block(self, symbol, interval=Interval.DAILY,
                       start_date=None, end_date=None, **kwargs) -> Optional[DataBlock]:
        path = self._F_PARQUET_PATTERN.format(symbol, interval.name)
        if not Path(path).exists():
            LOGGER.error(f"Could not find {path}")
            return None
        df = pd.read_parquet(path)
        # TODO interval 不一样 格式也应该不一样 这里datetime的格式先简化写了
        if start_date is not None or end_date is not None:
            if start_date is None:
                start_date = "1980-01-01"
            if end_date is None:
                end_date = datetime.today().strftime('%Y-%m-%d')
            df = df.loc[start_date:end_date]
        ret = DataBlock(symbol=symbol, interval=interval, data=df)
        return ret

    def save_data(self, data_block: DataBlock, **kwargs):
        path = self._F_PARQUET_PATTERN.format(data_block.symbol, data_block.interval.name)
        if Path(path).exists():
            LOGGER.warn(f"{path} already exists")
            return
        data_block.data.to_parquet(path, engine="pyarrow")

    def update_data(self, data_block: DataBlock, **kwargs):
        # TODO
        pass
