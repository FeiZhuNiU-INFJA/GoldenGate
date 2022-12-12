import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parents[1]))
from data.tushare_api import get_symbols, get_trade_cal, get_symbol_hist
from common.config import *
from data.base import *
from functools import partial
from p_tqdm import p_umap
start_date = "20000101"
retry = 3
def download_hist_data(ts_code, config: AssetConfig, retry=3):
    if retry==0:
        return
    try:
        df = get_symbol_hist(ts_code=ts_code, start_date=start_date)
        df.set_index("trade_date", inplace=True)
        df.sort_index(inplace=True)
        df.to_csv(config.DIR_HIST / f"{ts_code}.csv")
    except:
        print(ts_code)
        print(df.head())
        download_hist_data(ts_code, config, retry=retry-1)

def init_stock_SSE():
    LOGGER.info("Get SSE symbols")
    df_symbol = get_symbols(exchange=Exchange.SSE)
    df_symbol.to_csv(SSEConfig.F_SYMBOL)
    LOGGER.info("Get SSE trading calendar")
    df_trade_cal = get_trade_cal(exchange=Exchange.SSE)
    df_trade_cal.to_csv(SSEConfig.F_TRADE_CALENDAR)

    p_umap(partial(download_hist_data, config=SSEConfig), df_symbol.ts_code, num_cpus=4, desc="get history data")



def init_stock_SZSE():
    LOGGER.info("Get SZSE symbols")
    df_symbol = get_symbols(exchange=Exchange.SZSE)
    df_symbol.to_csv(SZSEConfig.F_SYMBOL)
    LOGGER.info("Get SZSE trading calendar")
    df_trade_cal = get_trade_cal(exchange=Exchange.SZSE)
    df_trade_cal.to_csv(SZSEConfig.F_TRADE_CALENDAR)

    p_umap(partial(download_hist_data, config=SZSEConfig), df_symbol.ts_code, num_cpus=4, desc="get history data")



if __name__ == '__main__':
    # 下载沪深、北交股票代码信息、交易日信息
    init_stock_SSE()
    init_stock_SZSE()
