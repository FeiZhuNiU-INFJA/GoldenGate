from data.base import *
from data.config import *


def get_stock_df(f_stock, date_str: str = None, strict=True, interval: int = None):
    """
    @date_str: 该时间点之前的数据 “2000-01-01”, 
    @strict: 这一天的数据是否必须存在, 不存在且strict=True的话, 不返回任何内容
    @interval: date_str之前多少个数据  (包括date_str)
    @annotation: 用于拿标签数据
    """
    df = pd.read_csv(f_stock)
    df.set_index("trade_date", inplace=True)
    df.index = pd.to_datetime(df.index, format='%Y%m%d')

    if date_str is not None:
        if strict and len(df[date_str:date_str]) == 0:
            return None
        df = df[:date_str]

    if interval is not None:
        df = df[-interval:]
        if len(df) != interval:
            return None
    return df


def get_df_symbols(market: Market = None):
    df = pd.read_csv(FILE_SYMBOLS_CN)
    if market is not None:
        df = df[df["market"] == market]
    return df


if __name__ == "__main__":
    print(get_df_symbols(market=Market.ZB).head())

