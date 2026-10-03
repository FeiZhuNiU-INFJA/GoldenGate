"""Market data clients: Baostock (CN) + yfinance (HK / US)."""
from __future__ import annotations

import logging
import os
import subprocess
import threading
import time
from pathlib import Path
from typing import Optional

import pandas as pd

from data.schema import normalize_bars

logger = logging.getLogger(__name__)

_bs_lock = threading.Lock()
_bs_logged_in = False
_bs_disabled_reason: Optional[str] = None
# py_mini_racer (used by akshare JS paths) is process-global and not thread-safe.
_ak_lock = threading.Lock()


def _cn_force_sina() -> bool:
    return os.environ.get("EXTREME_QUANT_CN_SINA", "").strip() in {"1", "true", "TRUE", "yes"} or bool(
        _bs_disabled_reason
    )


def _with_akshare(fn, *args, **kwargs):
    """Serialize all akshare calls to avoid mini_racer fatal crashes under threads."""
    with _ak_lock:
        return fn(*args, **kwargs)


def _to_iso(date_str: str) -> str:
    """YYYYMMDD or YYYY-MM-DD -> YYYY-MM-DD."""
    s = str(date_str).replace("-", "")[:8]
    return f"{s[:4]}-{s[4:6]}-{s[6:8]}"


def _ensure_baostock(retries: int = 3):
    global _bs_logged_in, _bs_disabled_reason
    import baostock as bs

    with _bs_lock:
        if _bs_disabled_reason:
            raise RuntimeError(f"baostock disabled: {_bs_disabled_reason}")
        if _bs_logged_in:
            return bs
        last_msg = ""
        for attempt in range(1, retries + 1):
            try:
                lg = bs.login()
            except Exception as exc:  # noqa: BLE001
                last_msg = str(exc)
                logger.warning("baostock login attempt %s/%s exception: %s", attempt, retries, last_msg)
                time.sleep(1.5 * attempt)
                continue
            if lg.error_code == "0":
                _bs_logged_in = True
                logger.info("baostock login ok")
                return bs
            last_msg = lg.error_msg or ""
            if "黑名单" in last_msg or "blacklist" in last_msg.lower():
                _bs_disabled_reason = last_msg
                logger.error("baostock blacklisted (%s); CN bars will use sina", last_msg)
                raise RuntimeError(f"baostock disabled: {last_msg}")
            logger.warning("baostock login attempt %s/%s failed: %s", attempt, retries, last_msg)
            time.sleep(1.5 * attempt)
        raise RuntimeError(
            f"baostock login failed after {retries} tries: {last_msg}. "
            "需要能直连 baostock 服务器（关掉 Clash TUN/系统代理后再试）"
        )


def _reset_baostock() -> None:
    """Drop a dead baostock session so the next call re-logins."""
    global _bs_logged_in
    import baostock as bs

    with _bs_lock:
        if not _bs_logged_in:
            return
        try:
            bs.logout()
        except Exception:  # noqa: BLE001
            pass
        _bs_logged_in = False


def _fetch_cn_bars_sina(raw_symbol: str, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
    """A-share daily via Sina HTTP JSON (no akshare/mini_racer — safe for threads)."""
    import json
    import urllib.error
    import urllib.request

    code = str(raw_symbol).zfill(6)
    prefix = "sh" if symbol.endswith(".SH") or code.startswith(("6", "9")) else "sz"
    # Sina returns the latest `datalen` daily bars. Size it to the requested
    # window (capped at 5000, ~20y) so incremental updates stay small.
    span_days = max(1, (pd.Timestamp(_to_iso(end_date)) - pd.Timestamp(_to_iso(start_date))).days + 1)
    datalen = min(5000, max(span_days, 5))
    url = (
        "https://money.finance.sina.com.cn/quotes_service/api/json_v2.php/"
        f"CN_MarketData.getKLineData?symbol={prefix}{code}&scale=240&ma=no&datalen={datalen}"
    )
    last_exc: Exception | None = None
    payload = ""
    for attempt in range(1, 4):
        try:
            req = urllib.request.Request(
                url,
                headers={"User-Agent": "Mozilla/5.0", "Referer": "https://finance.sina.com.cn"},
            )
            with urllib.request.urlopen(req, timeout=30) as resp:
                payload = resp.read().decode("utf-8", errors="ignore")
            break
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            last_exc = exc
            time.sleep(0.5 * attempt)
    else:
        raise RuntimeError(f"sina http failed for {symbol}: {last_exc}") from last_exc

    if not payload or payload in {"null", "[]"}:
        return normalize_bars(None)
    rows = json.loads(payload)
    if not rows:
        return normalize_bars(None)
    out = pd.DataFrame(rows)
    out = out.rename(
        columns={
            "day": "trade_date",
            "open": "open",
            "high": "high",
            "low": "low",
            "close": "close",
            "volume": "volume",
        }
    )
    out["amount"] = out["close"].astype(float) * out["volume"].astype(float)
    out["symbol"] = symbol
    out["market"] = "cn"
    out = normalize_bars(out)
    start = pd.Timestamp(_to_iso(start_date))
    end = pd.Timestamp(_to_iso(end_date))
    return out[(out["trade_date"] >= start) & (out["trade_date"] <= end)].reset_index(drop=True)


def _fetch_cn_benchmark_sina(start_date: str, end_date: str) -> pd.DataFrame:
    import akshare as ak

    df = _with_akshare(ak.stock_zh_index_daily, symbol="sh000300")
    if df is None or df.empty:
        raise RuntimeError("sina CSI300 empty")
    out = df.rename(
        columns={
            "date": "trade_date",
            "open": "open",
            "high": "high",
            "low": "low",
            "close": "close",
            "volume": "volume",
        }
    )
    if "amount" not in out.columns:
        out["amount"] = 0.0
    out["symbol"] = "000300.SH"
    out["market"] = "cn"
    out = normalize_bars(out)
    start = pd.Timestamp(_to_iso(start_date))
    end = pd.Timestamp(_to_iso(end_date))
    return out[(out["trade_date"] >= start) & (out["trade_date"] <= end)].reset_index(drop=True)


def _fetch_cn_universe_spot() -> pd.DataFrame:
    """Fallback universe via akshare stock_info (not eastmoney hist)."""
    import akshare as ak

    df = _with_akshare(ak.stock_info_a_code_name)
    df = df.rename(columns={"code": "code", "name": "name"})
    df["code"] = df["code"].astype(str).str.zfill(6)
    rows = []
    for _, r in df.iterrows():
        code = r["code"]
        if code.startswith(("8", "4", "92")):  # BJ / BSE
            continue
        if code.startswith(("6", "9")) and not code.startswith("92"):
            # SH main + STAR (60/68); skip odd 9xxxxx
            if not code.startswith(("60", "68")):
                continue
            sym = f"{code}.SH"
        elif code.startswith(("0", "3")):
            sym = f"{code}.SZ"
        else:
            continue
        rows.append({"symbol": sym, "name": r["name"], "market": "cn", "raw_symbol": code})
    return pd.DataFrame(rows).drop_duplicates("symbol").reset_index(drop=True)


def baostock_logout() -> None:
    global _bs_logged_in
    import baostock as bs

    with _bs_lock:
        if _bs_logged_in:
            bs.logout()
            _bs_logged_in = False


def _bs_code_to_symbol(code: str) -> Optional[str]:
    """sh.600000 / sz.000001 -> 600000.SH / 000001.SZ. Skip BJ."""
    code = str(code).strip().lower()
    if code.startswith("bj.") or code.startswith(("sh.8", "sz.8", "sh.4", "sz.4")):
        return None
    if code.startswith("sh."):
        num = code.split(".", 1)[1]
        if num.startswith(("8", "4")):
            return None
        return f"{num}.SH"
    if code.startswith("sz."):
        num = code.split(".", 1)[1]
        if num.startswith(("8", "4")):
            return None
        return f"{num}.SZ"
    return None


def _symbol_to_bs_code(symbol: str, raw_symbol: str) -> str:
    raw = str(raw_symbol).replace(".SH", "").replace(".SZ", "")
    if symbol.endswith(".SH") or raw.startswith("6") or raw.startswith("9"):
        return f"sh.{raw.zfill(6)}"
    return f"sz.{raw.zfill(6)}"


def _bs_result_to_df(rs) -> pd.DataFrame:
    rows = []
    while rs.error_code == "0" and rs.next():
        rows.append(rs.get_row_data())
    if not rows:
        return pd.DataFrame(columns=list(rs.fields))
    return pd.DataFrame(rows, columns=rs.fields)


def _yf_history(ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
    import yfinance as yf

    start = _to_iso(start_date)
    # yfinance end is exclusive-ish; bump one day
    end = (pd.Timestamp(_to_iso(end_date)) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    last_exc: Exception | None = None
    for attempt in range(1, 4):
        try:
            t = yf.Ticker(ticker)
            df = t.history(start=start, end=end, auto_adjust=True)
            if df is None or df.empty:
                return pd.DataFrame()
            out = df.reset_index()
            date_col = "Date" if "Date" in out.columns else out.columns[0]
            out = out.rename(
                columns={
                    date_col: "trade_date",
                    "Open": "open",
                    "High": "high",
                    "Low": "low",
                    "Close": "close",
                    "Volume": "volume",
                }
            )
            if "amount" not in out.columns:
                out["amount"] = out["close"].astype(float) * out["volume"].astype(float)
            time.sleep(0.25)
            return out
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            msg = str(exc).lower()
            if "rate" in msg or "too many" in msg:
                wait = 2.0 * attempt
                logger.warning("yfinance %s rate-limited (attempt %s); sleep %.1fs", ticker, attempt, wait)
                time.sleep(wait)
                continue
            logger.warning("yfinance %s failed: %s", ticker, exc)
            break
    raise RuntimeError(f"yfinance failed for {ticker}: {last_exc}") from last_exc


# ---------------------------------------------------------------------------
# Universes
# ---------------------------------------------------------------------------

def fetch_cn_universe() -> pd.DataFrame:
    try:
        bs = _ensure_baostock()
        rs = bs.query_stock_basic()
        raw = _bs_result_to_df(rs)
        if raw.empty:
            raise RuntimeError("baostock query_stock_basic returned empty")

        # Prefer listed A-shares: type=1 stock, status=1 listed when columns exist
        df = raw.copy()
        if "type" in df.columns:
            df = df[df["type"].astype(str) == "1"]
        if "status" in df.columns:
            df = df[df["status"].astype(str) == "1"]

        rows = []
        for _, r in df.iterrows():
            code = str(r.get("code", ""))
            sym = _bs_code_to_symbol(code)
            if sym is None:
                continue
            name = str(r.get("code_name", sym))
            raw_sym = sym.split(".")[0]
            rows.append({"symbol": sym, "name": name, "market": "cn", "raw_symbol": raw_sym})
        out = pd.DataFrame(rows).drop_duplicates("symbol").reset_index(drop=True)
        if out.empty:
            raise RuntimeError("cn universe empty after filtering")
        return out
    except Exception as exc:  # noqa: BLE001
        logger.warning("baostock universe failed (%s); falling back to akshare list", exc)
        return _fetch_cn_universe_spot()


def fetch_hk_universe() -> pd.DataFrame:
    """Hang Seng Index constituents (Wikipedia), fallback static HSI list."""
    try:
        import urllib.request

        req = urllib.request.Request(
            "https://en.wikipedia.org/wiki/Hang_Seng_Index",
            headers={"User-Agent": "extreme_quant/1.0"},
        )
        with urllib.request.urlopen(req, timeout=8) as resp:
            html = resp.read()
        tables = pd.read_html(html)
        cons = None
        for t in tables:
            cols = [str(c).lower() for c in t.columns]
            if any("ticker" in c or "symbol" in c or "code" in c for c in cols):
                cons = t
                break
            if any("company" in c for c in cols) and len(t) > 20:
                cons = t
                break
        if cons is None:
            raise RuntimeError("no HSI table")
        code_col = None
        for c in cons.columns:
            cl = str(c).lower()
            if "ticker" in cl or "symbol" in cl or cl == "code":
                code_col = c
                break
        name_col = None
        for c in cons.columns:
            if "company" in str(c).lower() or "name" in str(c).lower():
                name_col = c
                break
        rows = []
        if code_col is not None:
            for _, r in cons.iterrows():
                code = str(r[code_col]).strip().upper().replace(".HK", "")
                code = "".join(ch for ch in code if ch.isdigit())
                if not code:
                    continue
                code5 = code.zfill(5)
                name = str(r[name_col]) if name_col else code5
                rows.append(
                    {
                        "symbol": f"{code5}.HK",
                        "name": name,
                        "market": "hk",
                        "raw_symbol": code5,
                    }
                )
        if len(rows) >= 50:
            return pd.DataFrame(rows).drop_duplicates("symbol").reset_index(drop=True)
    except Exception as exc:  # noqa: BLE001
        logger.warning("HSI wikipedia universe failed: %s; using fallback list", exc)

    # Broad HSI-style mega/mid caps (not full official list when wiki blocked).
    tickers = [
        ("00001", "CK Hutchison"), ("00002", "CLP"), ("00003", "HK & China Gas"),
        ("00005", "HSBC"), ("00006", "Power Assets"), ("00011", "Hang Seng Bank"),
        ("00012", "Henderson Land"), ("00016", "SHK Properties"), ("00017", "New World Dev"),
        ("00027", "Galaxy Ent"), ("00066", "MTR"), ("00175", "Geely"),
        ("00267", "CITIC"), ("00288", "WH Group"), ("00316", "OOIL"),
        ("00386", "Sinopec"), ("00388", "HKEX"), ("00669", "Techtronic"),
        ("00688", "China Overseas"), ("00700", "Tencent"), ("00762", "China Unicom"),
        ("00823", "Link REIT"), ("00857", "PetroChina"), ("00868", "Xinyi Glass"),
        ("00883", "CNOOC"), ("00939", "CCB"), ("00941", "China Mobile"),
        ("00960", "Longfor"), ("00968", "Xinyi Solar"), ("00981", "SMIC"),
        ("00992", "Lenovo"), ("01024", "Kuaishou"), ("01088", "China Shenhua"),
        ("01109", "China Resources Land"), ("01113", "CK Asset"), ("01211", "BYD"),
        ("01299", "AIA"), ("01378", "China Hongqiao"), ("01398", "ICBC"),
        ("01810", "Xiaomi"), ("01876", "Budweiser APAC"), ("01928", "Sands China"),
        ("01997", "Wharf REIC"), ("02015", "Li Auto"), ("02020", "ANTA"),
        ("02269", "Wuxi Biologics"), ("02313", "Shenzhou Int"), ("02318", "Ping An"),
        ("02319", "Mengniu"), ("02331", "Li Ning"), ("02382", "Sunny Optical"),
        ("02388", "BOC Hong Kong"), ("02600", "Chalco"), ("02618", "JD Health"),
        ("02628", "China Life"), ("02688", "ENN Energy"), ("03690", "Meituan"),
        ("03750", "CATL-H"), ("03968", "CM Bank"), ("03988", "Bank of China"),
        ("03993", "CMOC"), ("06160", "BeOne Medicines"), ("06181", "Laopu Gold"),
        ("06618", "JD Logistics"), ("09618", "JD.com"), ("09626", "Bilibili"),
        ("09633", "Nongfu Spring"), ("09888", "Baidu"), ("09961", "Trip.com"),
        ("09988", "Alibaba"), ("09999", "NetEase"), ("02057", "ZTO Express"),
        ("02338", "Weichai Power"), ("01347", "Hua Hong Semi"), ("01519", "J&T Express"),
    ]
    return pd.DataFrame(
        [
            {"symbol": f"{c}.HK", "name": n, "market": "hk", "raw_symbol": c}
            for c, n in tickers
        ]
    ).drop_duplicates("symbol").reset_index(drop=True)


def fetch_us_universe() -> pd.DataFrame:
    """S&P 500 constituents from GitHub/Wikipedia, fallback mega-caps."""
    # Prefer GitHub CSV (often reachable when Wikipedia is blocked).
    try:
        import urllib.request

        url = "https://raw.githubusercontent.com/datasets/s-and-p-500-companies/main/data/constituents.csv"
        req = urllib.request.Request(url, headers={"User-Agent": "extreme_quant/1.0"})
        with urllib.request.urlopen(req, timeout=15) as resp:
            df = pd.read_csv(resp)
        sym_col = "Symbol" if "Symbol" in df.columns else df.columns[0]
        name_col = "Security" if "Security" in df.columns else ("Name" if "Name" in df.columns else df.columns[1])
        rows = []
        for _, r in df.iterrows():
            ticker = str(r[sym_col]).strip().upper().replace(".", "-")
            name = str(r[name_col])
            rows.append(
                {
                    "symbol": f"{ticker}.US",
                    "name": name,
                    "market": "us",
                    "raw_symbol": ticker,
                }
            )
        out = pd.DataFrame(rows).drop_duplicates("symbol").reset_index(drop=True)
        if len(out) > 100:
            logger.info("US universe from GitHub CSV: %s", len(out))
            return out
    except Exception as exc:  # noqa: BLE001
        logger.warning("S&P500 GitHub CSV failed: %s", exc)

    try:
        import urllib.request

        req = urllib.request.Request(
            "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
            headers={"User-Agent": "extreme_quant/1.0"},
        )
        with urllib.request.urlopen(req, timeout=8) as resp:
            html = resp.read()
        tables = pd.read_html(html)
        df = tables[0]
        sym_col = "Symbol" if "Symbol" in df.columns else df.columns[0]
        name_col = "Security" if "Security" in df.columns else df.columns[1]
        rows = []
        for _, r in df.iterrows():
            ticker = str(r[sym_col]).strip().upper().replace(".", "-")  # BRK.B -> BRK-B for yfinance
            name = str(r[name_col])
            rows.append(
                {
                    "symbol": f"{ticker}.US",
                    "name": name,
                    "market": "us",
                    "raw_symbol": ticker,
                }
            )
        out = pd.DataFrame(rows).drop_duplicates("symbol").reset_index(drop=True)
        if len(out) > 100:
            return out
    except Exception as exc:  # noqa: BLE001
        logger.warning("S&P500 wikipedia failed: %s; using fallback", exc)

    tickers = [
        "AAPL", "MSFT", "AMZN", "NVDA", "GOOGL", "META", "BRK-B", "LLY", "AVGO", "JPM",
        "TSLA", "UNH", "XOM", "V", "MA", "PG", "JNJ", "HD", "COST", "ABBV",
        "WMT", "MRK", "NFLX", "CRM", "BAC", "KO", "PEP", "TMO", "ORCL", "AMD",
        "CSCO", "ACN", "ABT", "MCD", "LIN", "DHR", "TXN", "WFC", "INTC", "PM",
        "DIS", "INTU", "AMGN", "CAT", "IBM", "GE", "QCOM", "VZ", "CMCSA", "NOW",
    ]
    return pd.DataFrame(
        {
            "symbol": [f"{t}.US" for t in tickers],
            "name": tickers,
            "market": "us",
            "raw_symbol": tickers,
        }
    )


# ---------------------------------------------------------------------------
# Bars
# ---------------------------------------------------------------------------

def fetch_cn_bars(raw_symbol: str, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
    if _cn_force_sina():
        return _fetch_cn_bars_sina(raw_symbol, symbol, start_date, end_date)
    try:
        bs = _ensure_baostock()
        code = _symbol_to_bs_code(symbol, raw_symbol)
        rs = bs.query_history_k_data_plus(
            code,
            "date,code,open,high,low,close,volume,amount",
            start_date=_to_iso(start_date),
            end_date=_to_iso(end_date),
            frequency="d",
            adjustflag="2",  # 前复权
        )
        if rs.error_code != "0":
            raise RuntimeError(f"baostock bars {code}: {rs.error_msg}")
        raw = _bs_result_to_df(rs)
        if raw.empty:
            return normalize_bars(None)
        out = raw.rename(
            columns={
                "date": "trade_date",
                "open": "open",
                "high": "high",
                "low": "low",
                "close": "close",
                "volume": "volume",
                "amount": "amount",
            }
        )
        out["symbol"] = symbol
        out["market"] = "cn"
        return normalize_bars(out)
    except Exception as exc:  # noqa: BLE001
        logger.warning("baostock bars failed for %s (%s); trying sina", symbol, exc)
        _reset_baostock()
        return _fetch_cn_bars_sina(raw_symbol, symbol, start_date, end_date)


def _hk_yf_ticker(raw_symbol: str) -> str:
    digits = "".join(ch for ch in str(raw_symbol) if ch.isdigit())
    return f"{int(digits):04d}.HK"


def fetch_hk_bars(raw_symbol: str, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
    skip_yf = os.environ.get("EQ_SKIP_YFINANCE", "").strip() in {"1", "true", "TRUE", "yes"}
    if not skip_yf:
        try:
            ticker = _hk_yf_ticker(raw_symbol)
            raw = _yf_history(ticker, start_date, end_date)
            if not raw.empty:
                raw["symbol"] = symbol
                raw["market"] = "hk"
                return normalize_bars(raw)
        except Exception as exc:  # noqa: BLE001
            logger.warning("yfinance hk bars failed %s (%s); trying sina", symbol, exc)

    import akshare as ak

    code = "".join(ch for ch in str(raw_symbol) if ch.isdigit()).zfill(5)
    df = _with_akshare(ak.stock_hk_daily, symbol=code, adjust="qfq")
    if df is None or df.empty:
        return normalize_bars(None)
    out = df.rename(
        columns={
            "date": "trade_date",
            "open": "open",
            "high": "high",
            "low": "low",
            "close": "close",
            "volume": "volume",
            "amount": "amount",
        }
    )
    if "amount" not in out.columns:
        out["amount"] = 0.0
    out["symbol"] = symbol
    out["market"] = "hk"
    out = normalize_bars(out)
    start, end = pd.Timestamp(_to_iso(start_date)), pd.Timestamp(_to_iso(end_date))
    return out[(out["trade_date"] >= start) & (out["trade_date"] <= end)].reset_index(drop=True)


def fetch_us_bars(raw_symbol: str, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
    ticker = str(raw_symbol).upper().replace(".US", "")
    skip_yf = os.environ.get("EQ_SKIP_YFINANCE", "").strip() in {"1", "true", "TRUE", "yes"}
    if not skip_yf:
        try:
            raw = _yf_history(ticker, start_date, end_date)
            if not raw.empty:
                raw["symbol"] = symbol
                raw["market"] = "us"
                return normalize_bars(raw)
        except Exception as exc:  # noqa: BLE001
            logger.warning("yfinance us bars failed %s (%s); trying sina", symbol, exc)

    import akshare as ak

    # Sina uses BF.B / BRK.B; yfinance uses BF-B / BRK-B.
    sina_ticker = ticker.replace("-", ".")
    candidates = [sina_ticker, ticker.replace("-", ""), ticker]
    df = None
    last_exc: Exception | None = None
    for cand in candidates:
        try:
            df = _with_akshare(ak.stock_us_daily, symbol=cand, adjust="qfq")
            if df is not None and not df.empty:
                break
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            df = None
    if df is None or df.empty:
        if last_exc:
            raise RuntimeError(f"us bars failed for {ticker}: {last_exc}") from last_exc
        return normalize_bars(None)
    out = df.rename(
        columns={
            "date": "trade_date",
            "open": "open",
            "high": "high",
            "low": "low",
            "close": "close",
            "volume": "volume",
            "amount": "amount",
        }
    )
    if "amount" not in out.columns:
        out["amount"] = 0.0
    out["symbol"] = symbol
    out["market"] = "us"
    out = normalize_bars(out)
    start, end = pd.Timestamp(_to_iso(start_date)), pd.Timestamp(_to_iso(end_date))
    return out[(out["trade_date"] >= start) & (out["trade_date"] <= end)].reset_index(drop=True)


def fetch_cn_benchmark(start_date: str, end_date: str) -> pd.DataFrame:
    try:
        bs = _ensure_baostock()
        rs = bs.query_history_k_data_plus(
            "sh.000300",
            "date,code,open,high,low,close,volume,amount",
            start_date=_to_iso(start_date),
            end_date=_to_iso(end_date),
            frequency="d",
            adjustflag="3",  # 不复权 for index
        )
        if rs.error_code != "0":
            raise RuntimeError(f"baostock CSI300: {rs.error_msg}")
        raw = _bs_result_to_df(rs)
        if raw.empty:
            raise RuntimeError("empty CSI300 from baostock")
        out = raw.rename(
            columns={
                "date": "trade_date",
                "open": "open",
                "high": "high",
                "low": "low",
                "close": "close",
                "volume": "volume",
                "amount": "amount",
            }
        )
        out["symbol"] = "000300.SH"
        out["market"] = "cn"
        return normalize_bars(out)
    except Exception as exc:  # noqa: BLE001
        logger.warning("baostock CSI300 failed (%s); trying sina", exc)
        return _fetch_cn_benchmark_sina(start_date, end_date)


def fetch_hk_benchmark(start_date: str, end_date: str) -> pd.DataFrame:
    try:
        raw = _yf_history("^HSI", start_date, end_date)
        if not raw.empty:
            raw["symbol"] = "HSI.HK"
            raw["market"] = "hk"
            return normalize_bars(raw)
    except Exception as exc:  # noqa: BLE001
        logger.warning("yfinance HSI failed (%s); trying sina", exc)

    import akshare as ak

    df = _with_akshare(ak.stock_hk_index_daily_sina, symbol="HSI")
    if df is None or df.empty:
        raise RuntimeError("empty HSI from sina")
    out = df.rename(
        columns={
            "date": "trade_date",
            "open": "open",
            "high": "high",
            "low": "low",
            "close": "close",
            "volume": "volume",
            "amount": "amount",
        }
    )
    if "amount" not in out.columns:
        out["amount"] = 0.0
    out["symbol"] = "HSI.HK"
    out["market"] = "hk"
    out = normalize_bars(out)
    start, end = pd.Timestamp(_to_iso(start_date)), pd.Timestamp(_to_iso(end_date))
    return out[(out["trade_date"] >= start) & (out["trade_date"] <= end)].reset_index(drop=True)


def fetch_us_benchmark(start_date: str, end_date: str) -> pd.DataFrame:
    try:
        raw = _yf_history("^GSPC", start_date, end_date)
        if not raw.empty:
            raw["symbol"] = "SPX.US"
            raw["market"] = "us"
            return normalize_bars(raw)
    except Exception as exc:  # noqa: BLE001
        logger.warning("yfinance SPX failed (%s); trying sina", exc)

    import akshare as ak

    df = _with_akshare(ak.index_us_stock_sina, symbol=".INX")
    if df is None or df.empty:
        raise RuntimeError("empty SPX from sina")
    out = df.rename(
        columns={
            "date": "trade_date",
            "open": "open",
            "high": "high",
            "low": "low",
            "close": "close",
            "volume": "volume",
            "amount": "amount",
        }
    )
    if "amount" not in out.columns:
        out["amount"] = 0.0
    out["symbol"] = "SPX.US"
    out["market"] = "us"
    out = normalize_bars(out)
    start, end = pd.Timestamp(_to_iso(start_date)), pd.Timestamp(_to_iso(end_date))
    return out[(out["trade_date"] >= start) & (out["trade_date"] <= end)].reset_index(drop=True)


_NDX_LIST_HTML = Path("/tmp/ndx_list.html")
_NDX_WIKI = "https://en.wikipedia.org/wiki/List_of_NASDAQ-100_companies"


def fetch_ndx_universe() -> pd.DataFrame:
    """Current Nasdaq-100 constituents from the Wikipedia list."""
    if not _NDX_LIST_HTML.exists() or _NDX_LIST_HTML.stat().st_size < 1000:
        subprocess.run(
            ["curl", "-fsSL", "-A", "Mozilla/5.0", "-o", str(_NDX_LIST_HTML), _NDX_WIKI],
            check=True,
        )
    table = pd.read_html(_NDX_LIST_HTML)[0]
    raw = table["Ticker"].astype(str).str.strip().str.upper().str.replace(".", "-", regex=False)
    frame = pd.DataFrame(
        {
            "symbol": raw.map(lambda ticker: f"{ticker}.US"),
            "name": table["Company"].astype(str),
            "market": "ndx",
            "raw_symbol": raw,
        }
    )
    return frame.drop_duplicates("symbol").reset_index(drop=True)


def fetch_ndx_bars(raw_symbol: str, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
    df = fetch_us_bars(raw_symbol, symbol, start_date, end_date)
    if df is None or df.empty:
        return df
    df = df.copy()
    df["market"] = "ndx"
    return df


def fetch_ndx_benchmark(start_date: str, end_date: str) -> pd.DataFrame:
    import akshare as ak

    df = _with_akshare(ak.index_us_stock_sina, symbol=".NDX")
    if df is None or df.empty:
        raise RuntimeError("empty Nasdaq-100 from sina")
    out = df.rename(
        columns={
            "date": "trade_date",
            "open": "open",
            "high": "high",
            "low": "low",
            "close": "close",
            "volume": "volume",
            "amount": "amount",
        }
    )
    if "amount" not in out.columns:
        out["amount"] = 0.0
    out["symbol"] = "NDX.US"
    out["market"] = "ndx"
    out = normalize_bars(out)
    start, end = pd.Timestamp(_to_iso(start_date)), pd.Timestamp(_to_iso(end_date))
    return out[(out["trade_date"] >= start) & (out["trade_date"] <= end)].reset_index(drop=True)


FETCH_UNIVERSE = {
    "cn": fetch_cn_universe,
    "hk": fetch_hk_universe,
    "us": fetch_us_universe,
    "ndx": fetch_ndx_universe,
}

FETCH_BARS = {
    "cn": fetch_cn_bars,
    "hk": fetch_hk_bars,
    "us": fetch_us_bars,
    "ndx": fetch_ndx_bars,
}

FETCH_BENCHMARK = {
    "cn": fetch_cn_benchmark,
    "hk": fetch_hk_benchmark,
    "us": fetch_us_benchmark,
    "ndx": fetch_ndx_benchmark,
}
