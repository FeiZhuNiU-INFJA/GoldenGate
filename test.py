from typing import Tuple
import numpy as np
import torch
import pandas as pd
from data.dataset import SingleSymbolDataset

# import config
from hubs import anno4, model4, model3
from config import BASE_FEATURES, DIR_DATA_HIST_CN
from tqdm import tqdm

import plotly.subplots as sp
import plotly.graph_objects as go

if __name__ == '__main__':
    # f_stock = "600000.SH.csv"
    f_stock = f"{DIR_DATA_HIST_CN}/603160.SH.csv"
    df = pd.read_csv(f_stock)
    df.set_index("trade_date", inplace=True)
    df.index = pd.to_datetime(df.index, format='%Y%m%d')
    df = df["20220601":]

    df = df[BASE_FEATURES]
    _data = df.copy()
    

    confs = [0] * len(df)
    types = ["Nan" for _ in range(len(df))] 
    for idx in tqdm(range(len(_data) - 128)):
        x = _data[idx:idx + 128].values
        x = SingleSymbolDataset.preprocess(x)
        x = torch.tensor(x).unsqueeze(dim=0).float()
        conf, clz = model4.inference(input_data=x)

        # score = {1: 1, 2: -1, 0: 0}.get(clz[0]), conf[0]
        
        types[idx+128] = {1: "Buy", 2: "Sell", 0: "Nan"}.get(clz[0])
        confs[idx+128] = conf[0]

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
        'Date': df.index,
        'Type': types,
        'Confidence': confs
    })

    for index, row in buy_sell_data.iterrows():
        if row['Type'] == 'Buy':
            color = 'red'
        elif row['Type'] == 'Sell':
            color = 'green'
        else:
            color = 'yellow'
        
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