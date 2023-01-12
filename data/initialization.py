import sys
from pathlib import Path
from typing import Type

sys.path.append(str(Path(__file__).parents[1]))
from data.tushare_api import get_symbols, get_trade_cal, get_symbol_hist
from common.config import *
from data.base import *
from functools import partial
from p_tqdm import p_umap

# TODO 每次只能取6000条数据  未来可能有问题
start_date = "20000101"


def download_hist_data(ts_code, config: Type[AssetConfig], retry=3):
    """
    :param ts_code:
    :param config:  决定了要保存到什么地方
    :param retry:
    :return:
    """
    if retry == 0:
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


def extra_works(df_symbol: DataFrame, config: Type[AssetConfig]):
    LOGGER.info("save industry info")
    industries = df_symbol.industry.value_counts().index.to_list()
    f_industry = config.HOME / "industry.csv"
    if f_industry.exists():
        with open(f_industry, 'r') as f:
            industries_exist = [l.strip() for l in f.readlines()]
    else:
        industries_exist = []
    with open(f_industry, 'a') as f:
        for industry in industries:
            if industry not in industries_exist:
                f.write(f"{industry}\n")
    

def init_stock_SSE():
    LOGGER.info("Get SSE symbols")
    df_symbol = get_symbols(exchange=Exchange.SSE)
    df_symbol.to_csv(SSEConfig.F_SYMBOL, index=False)

    LOGGER.info("Get SSE trading calendar")
    df_trade_cal = get_trade_cal(exchange=Exchange.SSE)
    df_trade_cal.to_csv(SSEConfig.F_TRADE_CALENDAR, index=False)

    LOGGER.info("do extra work")
    extra_works(df_symbol, SSEConfig)
    # # 获取2000年以后得日线数据
    p_umap(partial(download_hist_data, config=SSEConfig), df_symbol.ts_code, num_cpus=4, desc="get history data")


def init_stock_SZSE():
    LOGGER.info("Get SZSE symbols")
    df_symbol = get_symbols(exchange=Exchange.SZSE)
    df_symbol.to_csv(SZSEConfig.F_SYMBOL, index=False)

    LOGGER.info("Get SZSE trading calendar")
    df_trade_cal = get_trade_cal(exchange=Exchange.SZSE)
    df_trade_cal.to_csv(SZSEConfig.F_TRADE_CALENDAR, index=False)

    LOGGER.info("do extra work")
    extra_works(df_symbol, SZSEConfig)

    p_umap(partial(download_hist_data, config=SZSEConfig), df_symbol.ts_code, num_cpus=4, desc="get history data")


if __name__ == '__main__':
    # 下载沪深、北交股票代码信息、交易日信息
    init_stock_SSE()
    init_stock_SZSE()

    # # 补充
    # download_hist_data("688172.SH", config=SSEConfig)
    # download_hist_data("301265.SZ", config=SZSEConfig)

    # download_hist_data("600363.SH", config=SSEConfig)
    # download_hist_data("600785.SH", config=SSEConfig)
    # download_hist_data("001223.SZ", config=SZSEConfig)
