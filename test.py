from typing import Tuple
import numpy as np
import torch
import pandas as pd

# import config
from hubs import anno3, model3
from strategy.model import MyLSTM
from config import BASE_FEATURES, DIR_DATA_HIST_CN
# SEQ_LENGTH = 64
# HIDDEN_SIZE = 16
# NUM_LAYERS = 2
# DROPOUT = 0.1
# DIRECTIONS = 2
# IS_CLASSIFICATION = True
# device = torch.device("cpu")


if __name__ == '__main__':
    # f_stock = "600000.SH.csv"
    f_stock = f"{DIR_DATA_HIST_CN}/002230.SZ.csv"
    # f_stock = f"{DIR_DATA_HIST_CN}/003028.SZ.csv"
    # f_stock = f"{DIR_DATA_HIST_CN}/300573.SZ.csv"
    # f_stock = f"{DIR_DATA_HIST_CN}/600818.SH.csv"
    # f_stock = f"{DIR_DATA_HIST_CN}/688331.SH.csv"
    # f_stock = f"{DIR_DATA_HIST_CN}/601360.SH.csv"
    df = pd.read_csv(f_stock)
    df.set_index("trade_date", inplace=True)
    df.index = pd.to_datetime(df.index, format='%Y%m%d')
    df = df["20220601":]
    # df = df[BASE_FEATURES + [anno3.head_label]]  # trade_date 用于可视化

    # model = model3
    # result = model.test_model(
    #     data=df,
    #     threshold=None,
    #     head_label=anno3.head_label,
    #     head_features=BASE_FEATURES
    # )
    # result.to_csv("test.csv")
    # anno3.visualize(result)
    import plotly.graph_objects as go

    # 创建K线图
    fig = go.Figure(data=[go.Candlestick(x=df.index,
                    open=df['open'],
                    high=df['high'],
                    low=df['low'],
                    close=df['close'])])

    # 定义买卖点数据（示例数据）
    buy_points = [{'Date': '2023-01-10', 'Price': 150, 'Confidence': 0.8},
                {'Date': '2023-02-15', 'Price': 170, 'Confidence': 0.9}]

    sell_points = [{'Date': '2023-01-20', 'Price': 160, 'Confidence': 0.5},
                {'Date': '2023-03-05', 'Price': 180, 'Confidence': 0.85}]

    # 添加买卖点的散点图
    for point in buy_points:
        color = f'rgba(255, 0, 0, {point["Confidence"]})'  # 红色，透明度与置信度相关
        fig.add_trace(go.Scatter(x=[point['Date']], y=[point['Price']],
                                mode='markers',
                                marker=dict(color=color, size=10),
                                text=f"Buy (Confidence: {point['Confidence']})"))

    for point in sell_points:
        color = f'rgba(0, 128, 0, {point["Confidence"]})'  # 绿色，透明度与置信度相关
        fig.add_trace(go.Scatter(x=[point['Date']], y=[point['Price']],
                                mode='markers',
                                marker=dict(color=color, size=10),
                                text=f"Sell (Confidence: {point['Confidence']})"))

    # 显示图表
    fig.show()