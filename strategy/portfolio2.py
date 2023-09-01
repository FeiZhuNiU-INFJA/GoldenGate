import sys
from pathlib import Path
from typing import Union

sys.path.append(str(Path(__file__).parents[1]))
from data.dataset import SingleSymbolDataset
from functools import partial
from data.base import Market
from data.annotation import Annotation
import config
import hubs
from config import DIR_DATA_HIST_CN
from data.utils import get_stock_df, get_df_symbols
from strategy.model import MyLSTM, MyTransformer
import torch
from p_tqdm import p_map
import traceback
import pandas as pd
import tqdm


class ModelAnnotation(Annotation):

    @property
    def head_label(self):
        return "MyLSTM"

    def _labeling(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        model1 = hubs.model1
        model2 = hubs.model2
        head1 = "pred1"
        head2 = "pred2"
        data.set_index("trade_date", inplace=True)
        data.index = pd.to_datetime(data.index, format='%Y%m%d')
        data = model1.test_model(data, threshold=None, head_label=head1, head_features=config.BASE_FEATURES)
        data = model2.test_model(data, threshold=None, head_label=head2, head_features=config.BASE_FEATURES)
        data['result'] = (data[head1] + data[head2]) / 2.
        print(data.head())
        data.to_csv("test.csv")

        return data['result']

    def visualize(self, data_with_label: pd.DataFrame, **kwargs):
        pass


def heat(date_str, interval, model: Union[MyLSTM, MyTransformer], market: Market=None):
    """
    计算某一天某个市场的
    """
    def worker(f_csv):
        try:
            df = get_stock_df(f_stock=f_csv, date_str=date_str, interval=interval, strict=True)
            if df is None:
                return 0, 0
            df = df[config.BASE_FEATURES]
            x = df.values
            x = SingleSymbolDataset.preprocess(x)
            x = torch.tensor(x).unsqueeze(dim=0).float()
            conf, clz = model.inference(input_data=x)  # 1 buy -1 sell
            return {1: 1, 2: -1, 0: 0}.get(clz[0]), conf[0]
        except:
            print(traceback.format_exc())
            return 0, 0
    
    # 根据market过滤出股票
    df_symbols = get_df_symbols(market=market)
    ts_codes = df_symbols.ts_code.tolist()
    f_csvs = [f"{DIR_DATA_HIST_CN}/{ts_code}.csv" for ts_code in ts_codes]
    results = [worker(f_csv) for f_csv in tqdm.tqdm(f_csvs, desc="calculate heat")]
    # results = p_map(partial(worker), f_csvs, num_cpus=1, desc="calculate heat")
    return [(x[0], x[1][0]*x[1][1]) for x in zip(ts_codes, results)]  # code, clz*conf


def get_top_n_to_buy_sell(date_str, topN=20):
    # 获取所有股票代码
    df_symbols = get_df_symbols()
    df_symbols.set_index("ts_code", inplace=True)
    
    # 读取今天所有股票数据
    ensemble_scores = {}

    for model in [hubs.model3]:
        model.eval()
        scores = heat(date_str=date_str, interval=model.seq_length, model=model, market=None)
        for ts_code, score in scores:
            if ts_code not in ensemble_scores:
                ensemble_scores[ts_code] = [score]
            else:
                ensemble_scores[ts_code].append(score)
    # 所有模型结果取平均值
    ensemble_scores = [[key, sum(ensemble_scores[key]) / len(ensemble_scores[key])] for key in ensemble_scores]
    # 去掉ST的股票
    ensemble_scores = list(filter(lambda x: "ST" not in df_symbols.loc[x[0]]["name"], ensemble_scores))
    # 从小到大排序 小是卖 大是买
    ensemble_scores = sorted(ensemble_scores, key=lambda x: x[1])   

    print(ensemble_scores[-topN:][::-1])

    print(ensemble_scores[0:topN])


if __name__ == '__main__':
    # 找到买入和卖出信号最强的20只股票
    get_top_n_to_buy_sell("2023-08-31")
    #
    # df = pd.read_csv("/Users/yulin/workspace/extreme_quant/600000.SH.csv")
    # anno = ModelAnnotation()
    # df = anno.generate_data_with_label(df)
    # print(df.head())
