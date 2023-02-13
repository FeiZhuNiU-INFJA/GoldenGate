import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parents[1]))
from functools import partial
from data.base import Market
import config
from data.config import DIR_DATA_HIST_CN
from data.utils import get_stock_df, get_df_symbols
from strategy.model import MyLSTM
import torch
from p_tqdm import p_map
import traceback


def scores(date_str, interval, model: MyLSTM, market: Market=None):
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
            x[:, 0:4] /= x[0][0]
            x[:, 4] /= x[0][4]
            x[:, 5] /= x[0][5]
            x[:, :] -= 1
            x = torch.tensor(x).unsqueeze(dim=0).float()
            clz, conf = model.inference(input_data=x, threshold=None)  # 1 buy -1 sell
            return clz, conf
        except:
            print(traceback.format_exc())
            return 0, 0
    # 根据market过滤出股票

    df_symbols = get_df_symbols(market=market)
    ts_codes = df_symbols.ts_code.tolist()
    f_csvs = [f"{DIR_DATA_HIST_CN}/{ts_code}.csv" for ts_code in ts_codes]
    results = p_map(partial(worker), f_csvs, num_cpus=8, desc="calculate heat")
    return [(x[0], x[1][0], x[1][1]) for x in zip(ts_codes, results)]  # code, clz, conf


if __name__ == '__main__':
    # 载入模型
    SEQ_LENGTH = 64
    FEATURES_HEAD = config.BASE_FEATURES
    HIDDEN_SIZE = 16
    NUM_LAYERS = 2
    DROPOUT = 0.1
    DIRECTIONS = 2
    IS_CLASSIFICATION = True
    device = torch.device("cpu")
    model = MyLSTM(
        input_size=len(FEATURES_HEAD),
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        dropout_prob=DROPOUT,
        directions=DIRECTIONS,
        is_classification=IS_CLASSIFICATION,
        n_classes=config.anno1.n_class(),
        use_bceloss=False,
        device=device,
        seq_length=SEQ_LENGTH,
        weight="mylstm_last.pt",
    ).to(device)
    df_symbols = get_df_symbols()
    df_symbols.set_index("ts_code", inplace=True)
    # 读取今天所有股票数据
    scores = scores(date_str="20230213", interval=SEQ_LENGTH, model=model, market=None)
    scores = list(filter(lambda x: x[1] == 1, scores))
    scores = list(filter(lambda x: "ST" not in df_symbols.loc[x[0]]["name"], scores))
    scores = sorted(scores, key=lambda x: x[2], reverse=True)
    print(scores[0:20])


    # 模型推理

    # 找到买入和卖出信号最强的20只股票
    pass
