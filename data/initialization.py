import sys
import traceback
from pathlib import Path

sys.path.append(str(Path(__file__).parents[1]))
from data.tushare_api import *
from data.config import *
from data.base import *
from functools import partial
from p_tqdm import p_umap

# TODO 每次只能取6000条数据  未来可能有问题
start_date = "20000101"


def _download_hist_data(ts_code, retry=3, overwrite=True):
    """
    :param ts_code:
    :param retry:
    :return:
    """
    if retry == 0:
        print(ts_code)
        return
    try:
        if not overwrite and (DIR_DATA_HIST_CN / f"{ts_code}.csv").exists():
            return
        df = get_symbol_hist(ts_code=ts_code, start_date=start_date)
        df.set_index("trade_date", inplace=True)
        df.sort_index(inplace=True)
        df.to_csv(DIR_DATA_HIST_CN / f"{ts_code}.csv")

    except:
        # print(ts_code)
        # print(traceback.format_exc())
        _download_hist_data(ts_code, retry=retry - 1, overwrite=overwrite)


def extra_works(df_symbol: DataFrame):
    LOGGER.info("save industry info")
    industries = df_symbol.industry.value_counts().index.to_list()
    f_industry = STOCK_CN_HOME / "industry.csv"
    if f_industry.exists():
        with open(f_industry, 'r') as f:
            industries_exist = [l.strip() for l in f.readlines()]
    else:
        industries_exist = []
    with open(f_industry, 'a') as f:
        for industry in industries:
            if industry not in industries_exist:
                f.write(f"{industry}\n")

    LOGGER.info("save index info")  # 各种指数信息，下载下来之后根据需求再单独下载需要的指数k线
    df_indexes = get_index_base()
    df_indexes.to_csv(FILE_INDEX_CN, index=False)


def init_stock_CN():
    LOGGER.info("Get symbols")
    df_symbol = get_symbols()
    df_symbol.to_csv(FILE_SYMBOLS_CN, index=False)

    LOGGER.info("Get trading calendar")
    df_trade_cal = get_trade_cal()
    df_trade_cal.to_csv(FILE_TRADE_CALENDAR_CN, index=False)

    LOGGER.info("do extra work")
    extra_works(df_symbol)

    LOGGER.info("Get 上证指数")
    df = get_index("000001.SH")
    df.to_csv(STOCK_CN_HOME / "上证指数.csv", index=False)
    # 获取2000年以后得日线数据
    p_umap(partial(_download_hist_data, overwrite=True), df_symbol.ts_code, num_cpus=4, desc="get history data")


if __name__ == '__main__':
    init_stock_CN()
    # # 补充
    # _download_hist_data("000576.SZ")
    # _download_hist_data("000581.SZ")
    # _download_hist_data("000607.SZ")

