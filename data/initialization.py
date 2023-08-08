import sys
import traceback
from pathlib import Path

sys.path.append(str(Path(__file__).parents[1]))
from data.tushare_api import *
from config import *
from hubs import anno1, anno2, anno3
from data.base import *
from functools import partial
from p_tqdm import p_map
import numpy as np
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
        return 0
    try:
        if not overwrite and (DIR_DATA_HIST_CN / f"{ts_code}.csv").exists():
            return 1
        df = get_symbol_hist(ts_code=ts_code, start_date=start_date)
        df.set_index("trade_date", inplace=True)
        df.sort_index(inplace=True)
        df.to_csv(DIR_DATA_HIST_CN / f"{ts_code}.csv")
        return 1

    except:
        _download_hist_data(ts_code, retry=retry - 1, overwrite=overwrite)


def extra_works(df_symbol: DataFrame):
    LOGGER.info("保存产业、板块信息 (industry)")
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

    LOGGER.info("save index info (指数信息)")  # 各种指数信息，下载下来之后根据需求再单独下载需要的指数k线
    df_indexes = get_index_base()
    df_indexes.to_csv(FILE_INDEX_CN, index=False)


def init_stock_CN():
    LOGGER.info("Get symbols （股票代码）")
    df_symbol = get_symbols()
    df_symbol.to_csv(FILE_SYMBOLS_CN, index=False)

    LOGGER.info("Get trading calendar  （交易日信息）")
    df_trade_cal = get_trade_cal()
    df_trade_cal.to_csv(FILE_TRADE_CALENDAR_CN, index=False)

    LOGGER.info("do extra work")
    extra_works(df_symbol)

    LOGGER.info("Get 上证指数")
    df = get_index("000001.SH")
    df.to_csv(STOCK_CN_HOME / "上证指数.csv", index=False)
    # 获取2000年以后得日线数据
    targets = df_symbol.ts_code
    len_targets = len(targets)  # 剩余（待拉取）股票数量
    same_left_times = 0  # 同样剩余股票数的次数
    while len(targets) > 0:
        print(f"还剩: {len(targets)}, 剩余相同次数：{same_left_times}")
        if len(targets) == len_targets:
            same_left_times += 1
        else:
            same_left_times = 0
        len_targets = len(targets)
        # 如果尝试很多次还没有成功，则放弃
        if same_left_times == 20:
            break
        results = p_map(partial(_download_hist_data, overwrite=True),
                        targets,
                        num_cpus=16,
                        desc="get history data")
        print(results)
        targets = np.array(targets)[np.array(results) != 1].tolist()

def worker_anno_buysellpoint(f_csv):
    try:
        df = pd.read_csv(f_csv)

        for anno in [anno1, anno2, anno3]:
            df = anno.generate_data_with_label(df)
        if df is not None:
            df.to_csv(f_csv, index=False)
    except:
        print(f_csv, traceback.format_exc())


if __name__ == '__main__':
    # 下载日线数据
    init_stock_CN()
    # 万一有数据遗漏可以补充
    # _download_hist_data("000576.SZ")

    # 打标签
    CN_stocks = list(glob.glob(f"{DIR_DATA_HIST_CN}/*.csv"))
    p_umap(worker_anno_buysellpoint, CN_stocks, desc="Label Buy Sell Point", num_cpus=10)

