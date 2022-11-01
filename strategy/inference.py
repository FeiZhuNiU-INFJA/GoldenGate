from typing import Tuple
import numpy as np
import torch
import pandas as pd

device = torch.device("cpu")
p_model = "model.torchscript"
model = torch.jit.load(p_model)
model.eval()


def inference(data: pd.DataFrame) -> Tuple[int, float]:
    """
    :param data: t-n ~ t 的数据
    :return: (clz, confidence)
    clz: 0 其他  1 涨  2 跌
    """
    seq_length = 32   # 看前面多少个历史数据
    feature_length = 7  # 列数

    assert len(data) == seq_length
    assert len(data.columns) == feature_length
    assert data.columns.tolist() == ["open", "close", "high", "low", "volume", "number_of_trades", "taker_buy_volume"]

    # 数据处理
    data.loc[:, "open"] = np.log1p(data["open"])
    data.loc[:, "close"] = np.log1p(data["close"])
    data.loc[:, "high"] = np.log1p(data["high"])
    data.loc[:, "low"] = np.log1p(data["low"])
    data.loc[:, "volume"] = np.log1p(data["volume"])
    data.loc[:, "number_of_trades"] = np.log1p(data["number_of_trades"])
    data.loc[:, "taker_buy_volume"] = np.log1p(data["taker_buy_volume"])

    # 模型推理
    x = torch.tensor(data.values).unsqueeze(dim=0).float().to(device)
    num_layers = 2
    directions = 1
    batch_size = 1
    hidden_size = 16
    state_dim = (num_layers * directions, batch_size, hidden_size)
    h,c = torch.zeros(state_dim).to(device), torch.zeros(state_dim).to(device)
    output = model(x, (h, c))

    # 后处理
    output = torch.nn.Softmax(dim=1)(output)
    max_idx = torch.argmax(output).item()
    confidence = output[0][max_idx].item()
    return max_idx, confidence


if __name__ == '__main__':
    f_eth = "/Users/yulin/workspace/extreme_quant/data/raw_data/eth.parquet"
    df = pd.read_parquet(f_eth, engine="pyarrow")
    df = df[["open", "close", "high", "low", "volume", "number_of_trades", "taker_buy_volume"]]
    df.sort_index(inplace=True)
    #
    data = df.iloc[0:32]
    print(inference(data))