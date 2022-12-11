from data.tushare_api import get_symbols, get_trade_cal
from common.config import *
from data.base import *


def init_stock_sse():
    df_symbol = get_symbols(exchange=Exchange.SSE)
    df_symbol.to_csv(SSEConfig.F_SYMBOL)
    df_trade_cal = get_trade_cal(exchange=Exchange.SSE)
    df_trade_cal.to_csv(SSEConfig.F_TRADE_CALENDAR)


def init_stock_SZSE():
    df_symbol = get_symbols(exchange=Exchange.SZSE)
    df_symbol.to_csv(SZSEConfig.F_SYMBOL)
    df_trade_cal = get_trade_cal(exchange=Exchange.SZSE)
    df_trade_cal.to_csv(SZSEConfig.F_TRADE_CALENDAR)


if __name__ == '__main__':
    # 下载沪深、北交股票代码信息、交易日信息
    init_stock_sse()
    init_stock_SZSE()
