#!/usr/bin/env python3
"""Write a current industry label for each CN, HK, and US symbol.

CN is the CSRC industry from baostock. HK is Eastmoney's 所属行业.
US is the GICS sector of the S&P 500 from Wikipedia. None of the three is a
point-in-time history: a stock keeps today's industry on earlier dates.
"""
from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data import store
from data.akshare_client import _force_requests_direct, _maybe_clear_proxies
from data.market_client import _bs_code_to_symbol, _ensure_baostock, baostock_logout


def main() -> None:
    _maybe_clear_proxies()
    _force_requests_direct()
    written = {"cn": _cn(), "hk": _hk(), "us": _us()}
    for market, frame in written.items():
        path = store.market_dir(market) / "industry.parquet"
        frame.to_parquet(path, index=False)
        print(f"{market} industries={frame['industry'].nunique()} symbols={len(frame)} -> {path}")


def _cn() -> pd.DataFrame:
    bs = _ensure_baostock()
    rs = bs.query_stock_industry()
    if rs.error_code != "0":
        raise RuntimeError(rs.error_msg)
    rows = []
    while rs.error_code == "0" and rs.next():
        update_date, code, _name, industry, standard = rs.get_row_data()
        symbol = _bs_code_to_symbol(code)
        if symbol and str(industry).strip():
            rows.append(
                {
                    "symbol": symbol,
                    "industry": str(industry).strip(),
                    "source": standard,
                    "asof": update_date,
                }
            )
    baostock_logout()
    out = pd.DataFrame(rows).drop_duplicates("symbol")
    universe = set(store.read_universe("cn")["symbol"])
    return out.loc[out["symbol"].isin(universe)].reset_index(drop=True)


def _hk() -> pd.DataFrame:
    import akshare as ak

    universe = store.read_universe("hk")
    rows = []
    for raw, symbol in zip(universe["raw_symbol"], universe["symbol"]):
        code = str(raw).zfill(5)
        try:
            profile = ak.stock_hk_company_profile_em(code)
            industry = str(profile["所属行业"].iloc[0]).strip()
        except Exception as exc:  # noqa: BLE001
            print(f"hk {symbol} failed: {exc}")
            industry = ""
        if industry and industry.lower() != "nan":
            rows.append({"symbol": symbol, "industry": industry, "source": "eastmoney", "asof": ""})
        time.sleep(0.2)
    return pd.DataFrame(rows)


def _us() -> pd.DataFrame:
    html = Path("/tmp/sp500.html")
    if not html.exists():
        subprocess.run(
            [
                "curl",
                "-fsSL",
                "-A",
                "Mozilla/5.0",
                "-o",
                str(html),
                "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
            ],
            check=True,
        )
    table = pd.read_html(html)[0]
    table["raw_symbol"] = table["Symbol"].astype(str).str.replace(".", "-", regex=False)
    universe = store.read_universe("us")
    merged = universe.merge(
        table[["raw_symbol", "GICS Sector", "GICS Sub-Industry"]],
        on="raw_symbol",
        how="inner",
    )
    return pd.DataFrame(
        {
            "symbol": merged["symbol"],
            "industry": merged["GICS Sector"],
            "industry_detail": merged["GICS Sub-Industry"],
            "source": "wikipedia-gics",
            "asof": "",
        }
    )


if __name__ == "__main__":
    main()
