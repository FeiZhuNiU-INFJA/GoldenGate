from functools import partial
from data.base import Exchange
import config
from common.logger import LOGGER
from data.utils import get_stock_df
from strategy.model import MyLSTM
import torch
from p_tqdm import p_umap, p_map


def heat_score(date_str, interval, model: MyLSTM, exchange: Exchange):
    """
    计算某一天某个市场的热度
    """
    if exchange == Exchange.SSE:
        f_csvs = config.CSV_SSE_STOCKS
    elif exchange == Exchange.SZSE:
        f_csvs = config.CSV_SZSE_STOCKS
    else:
        LOGGER.warning("Unknown exchange")
        return
    
    

    def worker(f_csv):
        df = get_stock_df(f_stock=f_csv, date_str=date_str, interval=interval, strict=True)
        if df is None:
            return
        df = df[config.BASE_FEATURES]
        x = torch.tensor(df.values).unsqueeze(dim=0).float()
        clz, conf = model.inference(input_data=x, threshold=0.5) # 1 buy -1 sell
        return clz

    

    result = p_map(partial(worker), f_csvs,num_cpus=8, desc="calculate heat")
    n_to_buy = result.count(1)
    n_to_sell = result.count(-1)
    total = len(f_csvs)

    heat = (n_to_buy - n_to_sell) / total

    return heat


if __name__ == '__main__':
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
        weight="LSTM_seql64_best_baseline.pt",
    ).to(device)
    # 读取今天所有股票数据 
    heat = heat_score(date_str="2022-11-30", interval=SEQ_LENGTH, model=model, exchange=Exchange.SSE)
    print(heat)
    # 模型推理

    # 找到买入和卖出信号最强的20只股票
    pass
