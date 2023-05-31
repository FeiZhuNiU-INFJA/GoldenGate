from typing import Optional

import pandas as pd
import tushare as ts
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parents[1]))
from data.base import Interval, Exchange

ts.set_token("b9d8623c7dbde160e75147e2b47588810dd8dc3033ec8cadc9f64f53")
PRO = ts.pro_api(timeout=2)


def get_symbols(exchange: Optional[Exchange] = None):
    """
    拿到目前所有正常上市交易的股票列表
    :return:
    ts_code	    symbol	name	area industry	  fullname	            market	exchange	curr_type	list_status	list_date	 is_hs
0	600000.SH	600000	浦发银行	上海	银行	    上海浦东发展银行股份有限公司	    主板	    SSE	        CNY	        L	        19991110	     H
1	600004.SH	600004	白云机场	广东	机场	    广州白云国际机场股份有限公司	    主板	    SSE	        CNY	        L	        20030428	     N
2	600006.SH	600006	东风汽车	湖北	汽车整车	东风汽车股份有限公司	        主板	    SSE	        CNY	        L	        19990727	     N
3	600007.SH	600007	中国国贸	北京	园区开发	中国国际贸易中心股份有限公司	    主板	    SSE	        CNY	        L	        19990312	     H
4	600008.SH	600008	首创环保	北京	环境保护	北京首创生态环保集团股份有限公司	主板	    SSE	        CNY	        L	        20000427	     H
    """
    if exchange is None:
        exchange = ""
    data = PRO.stock_basic(exchange=exchange, list_status='L',
                           fields='ts_code,symbol,name,area,industry,list_date,list_status,fullname,exchange,market,is_hs,curr_type')
    return data


def get_trade_cal(exchange: Optional[Exchange] = None):
    """
    https://tushare.pro/document/2?doc_id=26
    看上去是一年更新一次，最早的从19901219开始
     '0'休市 '1'交易
     @param exchange 看上去只有SSE和SZSE有数据
    :return:
    	exchange	cal_date	is_open	pretrade_date
    0	SSE	        19901219	1	    None
    1	SSE	        19901220	1	    19901219
    2	SSE	        19901221	1	    19901220
    3	SSE	        19901222	0	    19901221
    4	SSE	        19901223	0	    19901221
    """
    if exchange is None:
        exchange = ""
    df = PRO.trade_cal(exchange=exchange, is_open="1", start_date='1989', end_date='2050')
    return df


def get_symbol_hist(ts_code, start_date=None, end_date=None, interval=Interval.DAILY):
    # TODO 每次最多20年的数据
    if interval == Interval.DAILY:
        # https://tushare.pro/document/2?doc_id=109
        df = ts.pro_bar(
            ts_code=ts_code, start_date=start_date, end_date=end_date,
            adj='qfq', asset='E', factors=['tor', 'vr'], adjfactor=True, retry_count=1
        )
    else:
        # https://tushare.pro/document/2?doc_id=27
        df = PRO.query(interval, ts_code=ts_code, start_date=start_date, end_date=end_date)
    return df


def get_index_base():
    """
    指数基本信息，查看有哪些指数
    https://tushare.pro/document/2?doc_id=94
    :return:
    """
    dfs = []
    for market in ["MSCI", "CSI", "SSE", "SZSE", "CICC", "SW", "OTH"]:
        dfs.append(PRO.index_basic(market=market))
    result = pd.concat(dfs, ignore_index=True)
    return result


def get_index(ts_code):
    """
    获取指数日线
    000001.SH 上证指数
    :param ts_code:
    :return:
    """
    df = PRO.index_daily(ts_code=ts_code)
    return df


if __name__ == '__main__':
    df = get_trade_cal()
    print(len(df))
    pass
    # df = get_symbol_hist("000002.SZ", start_date="19800101", end_date="19971230")
    # print(df.head())
    # print(len(df))
    # print(df.iloc[1690])
