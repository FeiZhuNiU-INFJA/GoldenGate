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
    import plotly.subplots as sp
    import plotly.graph_objects as go

    # 创建子图，一个用于K线图，一个用于柱状图
    fig = sp.make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05)

    # 添加K线图到第一个子图
    fig.add_trace(go.Candlestick(x=df.index,
                open=df['open'],
                high=df['high'],
                low=df['low'],
                close=df['close']), row=1, col=1)

    # 添加柱状图到第二个子图
    buy_sell_data = pd.DataFrame({
        'Date': ['2023-09-01', '2023-09-03', '2023-09-05'],
        'Type': ['Buy', 'Sell', 'Buy'],
        'Confidence': [0.1, 0.2, 0.3]
    })

    for index, row in buy_sell_data.iterrows():
        if row['Type'] == 'Buy':
            color = 'red'
        else:
            color = 'green'
        
        # 设置柱子的高度和颜色
        fig.add_trace(go.Bar(
            x=[row['Date']],
            y=[row['Confidence']],
            marker=dict(color=color),
            name=row['Type']
        ), row=2, col=1)
    # 自定义第二个子图的Y轴范围为0到1
    fig.update_yaxes(range=[0, 1], row=2, col=1)
    # 自定义图表布局
    fig.update_layout(
        xaxis_rangeslider_visible=False,  # 隐藏下方的时间范围滑块
        title='股票K线图与买卖点',
        xaxis_title='日期',
        yaxis_title='价格',
        height=600  # 增加图表的高度
    )

    fig.show()