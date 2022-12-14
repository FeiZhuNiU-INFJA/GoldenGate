from abc import ABCMeta, abstractmethod
from typing import Tuple, Type, Dict, List

import pandas as pd

from data.base import DataBlock, Interval
from data_old.database import DataSource
from data_old.indicator import Indicator
from data_old.labeling import Annotation


class FactorProcessor(metaclass=ABCMeta):
    """
    因子处理
    """

    @classmethod
    @abstractmethod
    def get_indicator_config(cls) -> Tuple[Type[Indicator], Dict, List[str]]:
        """
        配置indicators
        :return:
        """
        pass

    @classmethod
    def process_factors(cls, data: pd.DataFrame) -> pd.DataFrame:
        """
        对factor的额外处理
        :return:
        """
        return data


class TestFactorProcessor(FactorProcessor):

    @classmethod
    def get_indicator_config(cls) -> List[Tuple[Type[Indicator], Dict, List[str]]]:
        """
        :return:  [(indicator, indicator_params, indicator_names)]
        """
        return []

    @classmethod
    def process_factors(cls, data: pd.DataFrame):
        return data


class Dataloader(object):

    def __init__(self,
                 symbols: List[str],
                 database: Type[DataSource],
                 interval: Interval,
                 factor_processor: Type[FactorProcessor],
                 annotation_config: Tuple[Type[Annotation], dict],
                 start_date=None,
                 end_date=None) -> None:
        super().__init__()
        self.database = database
        self.interval = interval
        self.factor_processor: Type[FactorProcessor] = factor_processor
        self.annotation: Type[Annotation] = annotation_config[0]
        self.annotation_param: Dict = annotation_config[1]
        self.start_date = start_date
        self.end_date = end_date
        self.symbols = symbols
        self.data_blocks: List[DataBlock] = []

    def load_data(self):
        # TODO 并发
        for symbol in self.symbols:
            # 得到 OCHL数据
            data_block = self.database().get_data_block(symbol=symbol, interval=self.interval,
                                                        start_date=self.start_date, end_date=self.end_date)
            # 计算indicator 以及 labeling
            if data_block:
                for indicator, indicator_params, indicator_names in self.factor_processor.get_indicator_config():
                    results: List[pd.Series] = indicator.compute(data_block.data, **indicator_params)
                    assert len(results) == len(indicator_names)
                    for r, name in zip(results, indicator_names):
                        # if r.name in ["Close", "Open", "High", "Low", "Volume"]:
                        r.name = name
                        data_block.data = data_block.data.join(r)
                data_block.data = self.annotation.generate_labeled_data(data_block.data, **self.annotation_param)
                self.data_blocks.append(data_block)
        return self.data_blocks


