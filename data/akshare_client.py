"""Akshare wrappers with retry, sleep, and market-specific adapters."""
from __future__ import annotations

import logging
import os
import time
from typing import Callable, Optional

import pandas as pd
import requests

from config.settings import DOWNLOAD_MAX_RETRIES, DOWNLOAD_SLEEP_SEC
from data.schema import normalize_bars, rename_zh_columns

logger = logging.getLogger(__name__)

# Broken local proxies break Eastmoney calls; prefer direct access unless user opts in.
_PROXY_KEYS = (
    "http_proxy",
    "https_proxy",
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "all_proxy",
    "ALL_PROXY",
    "no_proxy",
    "NO_PROXY",
)


def _maybe_clear_proxies() -> None:
    """Clear env proxies. Note: macOS System Proxy is NOT in env — see _force_requests_direct."""
    if os.environ.get("EXTREME_QUANT_KEEP_PROXY", "").strip() in {"1", "true", "TRUE"}:
        return
    cleared = []
    for k in _PROXY_KEYS:
        if k in os.environ:
            cleared.append(k)
            os.environ.pop(k, None)
    # Make urllib/requests ignore residual proxy discovery where possible
    os.environ["NO_PROXY"] = "*"
    os.environ["no_proxy"] = "*"
    if cleared:
        logger.info("cleared proxy env vars: %s", ",".join(cleared))


def _force_requests_direct() -> None:
    """
    macOS System Proxy (Clash etc. at 127.0.0.1:7890) is picked up by
    urllib.request.getproxies() even when shell env is empty — that is why
    proxy_off alone often fails. Force requests/urllib to ignore proxies.
    """
    if os.environ.get("EXTREME_QUANT_KEEP_PROXY", "").strip() in {"1", "true", "TRUE"}:
        return

    # urllib.getproxies() on macOS reads System Configuration (scutil --proxy)
    try:
        import urllib.request

        urllib.request.getproxies = lambda: {}  # type: ignore[assignment]
        if hasattr(urllib.request, "getproxies_environment"):
            urllib.request.getproxies_environment = lambda: {}  # type: ignore[attr-defined]
    except Exception as exc:  # noqa: BLE001
        logger.warning("could not patch urllib.getproxies: %s", exc)

    if getattr(requests.sessions.Session.request, "_extreme_quant_direct", False):
        return

    _orig = requests.sessions.Session.request

    def _request(self, method, url, **kwargs):  # noqa: ANN001
        self.trust_env = False
        kwargs["proxies"] = {"http": None, "https": None}
        return _orig(self, method, url, **kwargs)

    _request._extreme_quant_direct = True  # type: ignore[attr-defined]
    requests.sessions.Session.request = _request  # type: ignore[method-assign]
    logger.info(
        "forced direct HTTP (bypassing macOS system proxy); "
        "set EXTREME_QUANT_KEEP_PROXY=1 to keep proxy"
    )


_maybe_clear_proxies()
_force_requests_direct()

CN_HIST_MAP = {
    "日期": "trade_date",
    "开盘": "open",
    "收盘": "close",
    "最高": "high",
    "最低": "low",
    "成交量": "volume",
    "成交额": "amount",
}

HK_HIST_MAP = CN_HIST_MAP.copy()
US_HIST_MAP = CN_HIST_MAP.copy()

INDEX_HIST_MAP = {
    "date": "trade_date",
    "日期": "trade_date",
    "open": "open",
    "开盘": "open",
    "close": "close",
    "收盘": "close",
    "high": "high",
    "最高": "high",
    "low": "low",
    "最低": "low",
    "volume": "volume",
    "成交量": "volume",
    "amount": "amount",
    "成交额": "amount",
}


def _sleep() -> None:
    time.sleep(DOWNLOAD_SLEEP_SEC)


def with_retry(fn: Callable, *args, retries: int = DOWNLOAD_MAX_RETRIES, **kwargs):
    last_exc: Optional[Exception] = None
    for attempt in range(1, retries + 1):
        try:
            result = fn(*args, **kwargs)
            _sleep()
            return result
        except Exception as exc:  # noqa: BLE001 - network / remote HTML errors
            last_exc = exc
            wait = DOWNLOAD_SLEEP_SEC * attempt * 2
            logger.warning("attempt %s/%s failed: %s; sleep %.1fs", attempt, retries, exc, wait)
            time.sleep(wait)
    raise RuntimeError(f"failed after {retries} retries: {last_exc}") from last_exc


def _import_ak():
    import akshare as ak

    return ak


# ---------------------------------------------------------------------------
# Universes
# ---------------------------------------------------------------------------

def fetch_cn_universe() -> pd.DataFrame:
    """A-share universe excluding Beijing Stock Exchange (codes starting with 8/4)."""
    ak = _import_ak()
    df = None
    errors: list[str] = []
    for fetcher in (
        lambda: ak.stock_info_a_code_name(),
        lambda: ak.stock_zh_a_spot_em(),
    ):
        try:
            raw = with_retry(fetcher)
            if raw is not None and not raw.empty:
                df = raw
                break
        except Exception as exc:  # noqa: BLE001
            errors.append(str(exc))
            df = None
    if df is None or df.empty:
        raise RuntimeError(f"unable to fetch CN universe: {errors}")

    # Normalize code/name columns across APIs
    rename = {}
    for a, b in (("code", "code"), ("代码", "code"), ("name", "name"), ("名称", "name")):
        if a in df.columns:
            rename[a] = b
    df = df.rename(columns=rename)
    if "code" not in df.columns:
        df = df.rename(columns={df.columns[0]: "code"})
    if "name" not in df.columns:
        df["name"] = df["code"]
    df = df[["code", "name"]].copy()
    df["code"] = df["code"].astype(str).str.zfill(6)

    def to_symbol(code: str) -> Optional[str]:
        if code.startswith(("8", "4")):  # BJ
            return None
        if code.startswith(("6", "9")):
            return f"{code}.SH"
        if code.startswith(("0", "3")):
            return f"{code}.SZ"
        return None

    df["symbol"] = df["code"].map(to_symbol)
    df = df.dropna(subset=["symbol"]).copy()
    df["market"] = "cn"
    df["raw_symbol"] = df["code"]
    return df[["symbol", "name", "market", "raw_symbol"]].reset_index(drop=True)


def fetch_hk_universe() -> pd.DataFrame:
    """Hang Seng Index constituents."""
    ak = _import_ak()
    cons = None
    for fetcher in (
        lambda: ak.index_stock_cons(symbol="HSI"),
        lambda: ak.stock_hk_index_spot_em(),  # fallback: later filter manually not ideal
    ):
        try:
            cons = with_retry(fetcher)
            if cons is not None and not cons.empty:
                break
        except Exception as exc:  # noqa: BLE001
            logger.warning("hk universe fetcher failed: %s", exc)
            cons = None

    if cons is None or cons.empty:
        # Hard fallback: well-known HSI names via hk spot + filter famous list if available
        try:
            famous = with_retry(ak.stock_hk_famous_spot_em)
            df = famous.rename(columns={"代码": "code", "名称": "name"})[["code", "name"]].copy()
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError("unable to fetch HK universe") from exc
    else:
        # index_stock_cons typically has 品种代码 / 品种名称
        rename = {}
        for a, b in (
            ("品种代码", "code"),
            ("成分券代码", "code"),
            ("代码", "code"),
            ("品种名称", "name"),
            ("成分券名称", "name"),
            ("名称", "name"),
        ):
            if a in cons.columns:
                rename[a] = b
        df = cons.rename(columns=rename)
        if "code" not in df.columns:
            # stock_hk_index_spot_em is index list, not constituents — use hk spot sample
            spot = with_retry(ak.stock_hk_spot_em)
            df = spot.rename(columns={"代码": "code", "名称": "name"})[["code", "name"]].head(80).copy()
        else:
            if "name" not in df.columns:
                df["name"] = df["code"]
            df = df[["code", "name"]].copy()

    df["code"] = df["code"].astype(str).str.replace(r"\.HK$", "", regex=True).str.zfill(5)
    df["symbol"] = df["code"] + ".HK"
    df["market"] = "hk"
    df["raw_symbol"] = df["code"]
    return df[["symbol", "name", "market", "raw_symbol"]].drop_duplicates("symbol").reset_index(drop=True)


def fetch_us_universe() -> pd.DataFrame:
    """S&P 500 constituents (best-effort via Akshare)."""
    ak = _import_ak()
    df = None
    errors: list[str] = []
    for fetcher in (
        lambda: ak.index_stock_cons(symbol="SPX"),
        lambda: ak.index_us_stock_sina(symbol=".INX"),
        lambda: ak.index_us_stock_sina(symbol="SPX"),
    ):
        try:
            raw = with_retry(fetcher)
            if raw is not None and not raw.empty:
                df = raw
                break
        except Exception as exc:  # noqa: BLE001
            errors.append(str(exc))

    if df is None or df.empty:
        # Fallback: liquid mega-caps so pipeline still works
        logger.warning("S&P500 constituents unavailable (%s); using mega-cap fallback", errors)
        tickers = [
            "AAPL", "MSFT", "AMZN", "NVDA", "GOOGL", "META", "BRK.B", "LLY", "AVGO", "JPM",
            "TSLA", "UNH", "XOM", "V", "MA", "PG", "JNJ", "HD", "COST", "ABBV",
        ]
        return pd.DataFrame(
            {
                "symbol": [f"{t}.US" for t in tickers],
                "name": tickers,
                "market": "us",
                "raw_symbol": tickers,
            }
        )

    rename = {}
    for a, b in (
        ("品种代码", "code"),
        ("code", "code"),
        ("代码", "code"),
        ("symbol", "code"),
        ("品种名称", "name"),
        ("name", "name"),
        ("名称", "name"),
    ):
        if a in df.columns:
            rename[a] = b
    out = df.rename(columns=rename)
    if "code" not in out.columns:
        out = out.rename(columns={out.columns[0]: "code"})
    if "name" not in out.columns:
        out["name"] = out["code"]
    out["code"] = out["code"].astype(str).str.upper().str.replace(r"\.US$", "", regex=True)
    # Eastmoney US hist often needs "105.AAPL" style; store ticker as raw and resolve later
    out["symbol"] = out["code"] + ".US"
    out["market"] = "us"
    out["raw_symbol"] = out["code"]
    return out[["symbol", "name", "market", "raw_symbol"]].drop_duplicates("symbol").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Bars
# ---------------------------------------------------------------------------

def _to_bars(df: pd.DataFrame, symbol: str, market: str, mapping: dict) -> pd.DataFrame:
    if df is None or df.empty:
        return normalize_bars(None)
    out = rename_zh_columns(df, mapping)
    # English fallbacks already partially covered
    for src, dst in INDEX_HIST_MAP.items():
        if src in out.columns and dst not in out.columns:
            out = out.rename(columns={src: dst})
    out["symbol"] = symbol
    out["market"] = market
    if "volume" not in out.columns:
        out["volume"] = 0.0
    if "amount" not in out.columns:
        out["amount"] = 0.0
    return normalize_bars(out)


def fetch_cn_bars(raw_symbol: str, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
    ak = _import_ak()
    code = raw_symbol.zfill(6)
    errors: list[str] = []

    # Primary: Eastmoney (1 try). Often empty-reply on some networks — fall through to Sina.
    try:
        df = ak.stock_zh_a_hist(
            symbol=code,
            period="daily",
            start_date=start_date,
            end_date=end_date,
            adjust="qfq",
        )
        out = _to_bars(df, symbol, "cn", CN_HIST_MAP)
        if not out.empty:
            return out
    except Exception as exc:  # noqa: BLE001
        errors.append(f"eastmoney:{exc}")
        logger.warning("eastmoney hist failed for %s, trying sina: %s", symbol, exc)

    # Fallback: Sina (symbol like sz000001 / sh600000)
    prefix = "sh" if code.startswith(("6", "9")) else "sz"
    try:
        df = with_retry(
            ak.stock_zh_a_daily,
            symbol=f"{prefix}{code}",
            start_date=start_date,
            end_date=end_date,
            adjust="qfq",
        )
        # sina columns: date, open, high, low, close, volume, amount, ...
        rename = {
            "date": "trade_date",
            "open": "open",
            "high": "high",
            "low": "low",
            "close": "close",
            "volume": "volume",
            "amount": "amount",
        }
        out = _to_bars(df.rename(columns=rename), symbol, "cn", {})
        if not out.empty:
            logger.info("cn bars via sina fallback for %s", symbol)
            return out
    except Exception as exc:  # noqa: BLE001
        errors.append(f"sina:{exc}")

    raise RuntimeError(f"cn bars failed for {symbol}: {errors}")


def fetch_hk_bars(raw_symbol: str, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
    ak = _import_ak()
    code = raw_symbol.zfill(5)
    df = with_retry(
        ak.stock_hk_hist,
        symbol=code,
        period="daily",
        start_date=start_date,
        end_date=end_date,
        adjust="qfq",
    )
    return _to_bars(df, symbol, "hk", HK_HIST_MAP)


def _resolve_us_em_symbol(ticker: str) -> str:
    """Map ticker to Eastmoney us symbol like '105.AAPL' when possible."""
    ak = _import_ak()
    ticker = ticker.upper()
    try:
        spot = with_retry(ak.stock_us_spot_em)
        code_col = "代码" if "代码" in spot.columns else spot.columns[0]
        name_col = "名称" if "名称" in spot.columns else None
        # codes look like 105.AAPL
        mask = spot[code_col].astype(str).str.endswith(f".{ticker}") | (
            spot[code_col].astype(str).str.upper() == ticker
        )
        if name_col is not None:
            mask = mask | (spot[name_col].astype(str).str.upper() == ticker)
        hits = spot.loc[mask]
        if not hits.empty:
            return str(hits.iloc[0][code_col])
    except Exception as exc:  # noqa: BLE001
        logger.debug("us spot resolve failed for %s: %s", ticker, exc)
    # Common NASDAQ prefix guess
    return f"105.{ticker}"


def fetch_us_bars(raw_symbol: str, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
    ak = _import_ak()
    ticker = raw_symbol.upper().replace(".US", "")
    em_symbol = _resolve_us_em_symbol(ticker)
    df = with_retry(
        ak.stock_us_hist,
        symbol=em_symbol,
        period="daily",
        start_date=start_date,
        end_date=end_date,
        adjust="qfq",
    )
    return _to_bars(df, symbol, "us", US_HIST_MAP)


# ---------------------------------------------------------------------------
# Benchmarks
# ---------------------------------------------------------------------------

def fetch_cn_benchmark(start_date: str, end_date: str) -> pd.DataFrame:
    ak = _import_ak()
    out = None
    errors: list[str] = []
    for fetcher in (
        lambda: ak.index_zh_a_hist(symbol="000300", period="daily", start_date=start_date, end_date=end_date),
        lambda: ak.stock_zh_index_daily_em(symbol="sh000300"),
        lambda: ak.stock_zh_index_daily(symbol="sh000300"),
    ):
        try:
            df = with_retry(fetcher)
            out = _to_bars(df, "000300.SH", "cn", INDEX_HIST_MAP)
            if not out.empty:
                break
        except Exception as exc:  # noqa: BLE001
            errors.append(str(exc))
            out = None
    if out is None or out.empty:
        raise RuntimeError(f"unable to fetch CSI300 history: {errors}")
    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)
    return out[(out["trade_date"] >= start) & (out["trade_date"] <= end)].reset_index(drop=True)


def fetch_hk_benchmark(start_date: str, end_date: str) -> pd.DataFrame:
    ak = _import_ak()
    out = None
    for kwargs in (
        {"symbol": "HSI"},
        {"symbol": "hkHSI"},
    ):
        try:
            df = with_retry(ak.stock_hk_index_daily_em, **kwargs)
            out = _to_bars(df, "HSI.HK", "hk", INDEX_HIST_MAP)
            if not out.empty:
                break
        except Exception as exc:  # noqa: BLE001
            logger.warning("hk benchmark %s failed: %s", kwargs, exc)
    if out is None or out.empty:
        # Last resort: use 00700 as proxy is wrong — raise
        raise RuntimeError("unable to fetch Hang Seng index history")
    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)
    return out[(out["trade_date"] >= start) & (out["trade_date"] <= end)].reset_index(drop=True)


def fetch_us_benchmark(start_date: str, end_date: str) -> pd.DataFrame:
    ak = _import_ak()
    out = None
    for symbol in (".INX", "SPX", "SP500"):
        try:
            df = with_retry(ak.index_us_stock_sina, symbol=symbol)
            out = _to_bars(df, "SPX.US", "us", INDEX_HIST_MAP)
            if not out.empty:
                break
        except Exception as exc:  # noqa: BLE001
            logger.warning("us benchmark %s failed: %s", symbol, exc)
    if out is None or out.empty:
        raise RuntimeError("unable to fetch S&P 500 history")
    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)
    return out[(out["trade_date"] >= start) & (out["trade_date"] <= end)].reset_index(drop=True)


def _ndx_fetch(name: str):
    from data import market_client

    return getattr(market_client, name)


FETCH_BARS = {
    "cn": fetch_cn_bars,
    "hk": fetch_hk_bars,
    "us": fetch_us_bars,
    "ndx": lambda *args, **kwargs: _ndx_fetch("fetch_ndx_bars")(*args, **kwargs),
}

FETCH_UNIVERSE = {
    "cn": fetch_cn_universe,
    "hk": fetch_hk_universe,
    "us": fetch_us_universe,
    "ndx": lambda *args, **kwargs: _ndx_fetch("fetch_ndx_universe")(*args, **kwargs),
}

FETCH_BENCHMARK = {
    "cn": fetch_cn_benchmark,
    "hk": fetch_hk_benchmark,
    "us": fetch_us_benchmark,
    "ndx": lambda *args, **kwargs: _ndx_fetch("fetch_ndx_benchmark")(*args, **kwargs),
}
