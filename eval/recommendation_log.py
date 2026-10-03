"""Cumulative close-to-close returns for a frozen list of recommendations."""
from __future__ import annotations

import numpy as np
import pandas as pd

HORIZONS = (5, 10, 20)

_US_BOOK = {
    "title": "美股推荐跟踪：三个模型前 5 名交集",
    "since": "2026-10-01",
    "models": "`checkpoints/ensemble/ranker_us_h{5,10,20}_s{0,1,2}.txt`",
    "bench": "标普 500",
    "ledger": "docs/live/us-intersection.json",
    "html": "docs/live/intersection.html",
}


def session_path(
    symbol_close: dict[str, pd.Series],
    bench: pd.Series,
    signal_date,
    symbols: list[str],
    max_hold: int = 20,
) -> pd.DataFrame:
    """Equal-weight cumulative return from the signal close, counted on the benchmark calendar.

    ``hold`` is the number of benchmark sessions after the signal date. A session is
    omitted when any name or the benchmark lacks a positive close, so a later row
    still keeps its original hold count. The signal session itself is hold 0.
    """
    columns = ["trade_date", "hold", *symbols, "port", "bench", "excess"]
    if not symbols:
        return pd.DataFrame(columns=columns)
    signal = pd.Timestamp(signal_date).normalize()
    bench = _positive(bench)
    if signal not in bench.index:
        return pd.DataFrame(columns=columns)
    prepared: dict[str, pd.Series] = {}
    entries: dict[str, float] = {}
    for symbol in symbols:
        series = _positive(symbol_close[symbol])
        if signal not in series.index:
            return pd.DataFrame(columns=columns)
        prepared[symbol] = series
        entries[symbol] = float(series.loc[signal])
    entry_bench = float(bench.loc[signal])
    calendar = bench.index[bench.index >= signal][: max_hold + 1]
    rows = []
    for hold, day in enumerate(calendar):
        per: dict[str, float] = {}
        complete = True
        for symbol in symbols:
            series = prepared[symbol]
            if day not in series.index:
                complete = False
                break
            per[symbol] = float(series.loc[day]) / entries[symbol] - 1.0
        if not complete:
            continue
        port = float(np.mean([per[symbol] for symbol in symbols]))
        bench_ret = float(bench.loc[day]) / entry_bench - 1.0
        rows.append(
            {
                "trade_date": day,
                "hold": hold,
                **per,
                "port": port,
                "bench": bench_ret,
                "excess": port - bench_ret,
            }
        )
    return pd.DataFrame(rows, columns=columns)


def horizon_state(path: pd.DataFrame, horizon: int, *, flat: bool) -> str | pd.Series:
    """``未到期`` / ``缺数据`` / ``—`` for an empty book, or the matching path row."""
    if flat:
        return "—"
    if path.empty or "hold" not in path.columns:
        return "缺数据"
    hit = path.loc[path["hold"] == horizon]
    if not hit.empty:
        return hit.iloc[0]
    if int(path["hold"].max()) < horizon:
        return "未到期"
    return "缺数据"


def render_log(
    signals: list[dict],
    paths: dict[str, pd.DataFrame],
    names: dict[str, str],
    entries: dict[str, dict[str, float]],
    book: dict | None = None,
) -> str:
    """Human-readable tracker. Tables are regenerated from the ledger and bars."""
    spec = _US_BOOK if book is None else book
    bench = spec["bench"]
    lines = [
        f"# {spec['title']}",
        "",
        (
            f"从 {spec['since']} 收盘开始记。标签分三档：前 5、第 6–15、其余。"
            f"每个期限用三个已经训好的模型（{spec['models']}，分数符号都是 +1）各自取前 5 名，"
            f"三个名单的交集就是当天的推荐。交集为空也记一行，当天没有持仓。"
            f"5 日、10 日、20 日各自一本账。早于 {spec['since']} 的交易日不补。"
        ),
        "",
        "行情更新之后运行：",
        "",
        "```bash",
        "python scripts/update_us_recommendations.py",
        "```",
        "",
        (
            "脚本把账本里还没有的新交易日补上，并按本地复权收盘重算下面的涨跌幅。"
            f"名单在 `{spec['ledger']}`。同一轮会重写 `{spec['html']}`，"
            "也会更新另一本账（标普 500 与纳斯达克 100）。"
            "手改本页或那个 HTML 会被下一次运行覆盖；想留一句话，写在对应信号的 `note` 字段。"
        ),
        "",
        "## 口径",
        "",
        (
            f"入场价是信号日收盘。持有 n 个交易日按{bench} 自己的交易日往后数，"
            f"涨跌幅 = 当天收盘 / 信号日收盘 − 1。组合是推荐名单等权。超额 = 组合涨跌幅 − 同期{bench} 涨跌幅。"
        ),
        "",
        (
            f"名单里有一只当天没有可用收盘，这一行就不写。持有天数仍按{bench} 的那一天计，"
            "所以后面的行不会把缺的那一天算进持有期。5 / 10 / 20 日三列就是持有天数走到 5、10、20 的那一行；"
            "还没走到写「未到期」。表记到持有 20 日为止。"
        ),
        "",
        "收盘价不是正数、相对前后约 11 日中位数偏离超过 5 倍、或单日涨跌超过 2.5 倍的打印，视为没有收盘。",
        "",
    ]
    for horizon in HORIZONS:
        lines.extend(["", f"## {horizon} 日模型", "", "### 总表", "", _summary_table(signals, paths, horizon), ""])
        for signal in signals:
            lines.extend(
                _signal_section(
                    signal,
                    horizon,
                    _frame(paths, signal["date"], horizon),
                    names,
                    _entry_map(entries, signal["date"], horizon),
                )
            )
    lines.append("")
    return "\n".join(lines)


def page_data(
    signals: list[dict],
    paths: dict[str, pd.DataFrame],
    names: dict[str, str],
    entries: dict[str, dict[str, float]],
) -> dict:
    """Serializable books of cumulative returns, one list per model horizon. Fractions, not percents."""
    groups = {str(horizon): [] for horizon in HORIZONS}
    as_of = ""
    for signal in signals:
        for horizon in HORIZONS:
            picks = list(_horizon_book(signal, horizon).get("picks") or [])
            path = _frame(paths, signal["date"], horizon)
            flat = not picks
            day_entries = _entry_map(entries, signal["date"], horizon)
            scores = _score_lookup(signal, horizon)
            pick_rows = []
            for symbol in picks:
                pick_rows.append(
                    {
                        "symbol": _code(symbol),
                        "name": names.get(symbol, ""),
                        "entry": _num(day_entries.get(symbol)),
                        "scores": [_num(scores.get(symbol, {}).get(seed)) for seed in (0, 1, 2)],
                    }
                )
            path_rows = []
            if not flat and not path.empty:
                for _, row in path.iterrows():
                    day = pd.Timestamp(row["trade_date"]).strftime("%Y-%m-%d")
                    path_rows.append(
                        {
                            "date": day,
                            "hold": int(row["hold"]),
                            "port": _num(row["port"]),
                            "bench": _num(row["bench"]),
                            "excess": _num(row["excess"]),
                            "names": {_code(symbol): _num(row[symbol]) for symbol in picks},
                        }
                    )
                    if day > as_of:
                        as_of = day
            holds = {}
            for hold in HORIZONS:
                state = horizon_state(path, hold, flat=flat)
                if isinstance(state, str):
                    holds[str(hold)] = {"status": state}
                else:
                    holds[str(hold)] = {
                        "status": "到期",
                        "port": _num(state["port"]),
                        "bench": _num(state["bench"]),
                        "excess": _num(state["excess"]),
                    }
            groups[str(horizon)].append(
                {
                    "date": signal["date"],
                    "note": (signal.get("note") or "").strip(),
                    "flat": flat,
                    "held": None if flat or path.empty else int(path["hold"].max()),
                    "picks": pick_rows,
                    "horizons": holds,
                    "path": path_rows,
                }
            )
    if not as_of and signals:
        as_of = max(item["date"] for item in signals)
    return {"as_of": as_of, "horizons": list(HORIZONS), "groups": groups}


def _num(value) -> float | None:
    if value is None or not np.isfinite(value):
        return None
    return round(float(value), 8)


def _horizon_book(signal: dict, horizon: int) -> dict:
    raw = signal.get("horizons") or {}
    book = raw.get(str(horizon)) or raw.get(horizon) or {}
    return book if isinstance(book, dict) else {}


def _frame(paths: dict, date: str, horizon: int) -> pd.DataFrame:
    day = paths.get(date, {})
    if isinstance(day, pd.DataFrame):
        return day
    if horizon in day:
        frame = day[horizon]
    elif str(horizon) in day:
        frame = day[str(horizon)]
    else:
        return pd.DataFrame()
    return frame if isinstance(frame, pd.DataFrame) else pd.DataFrame()


def _entry_map(entries: dict, date: str, horizon: int) -> dict[str, float]:
    day = entries.get(date, {})
    if not day:
        return {}
    sample = next(iter(day.values()))
    if not isinstance(sample, dict):
        return day
    found = day.get(horizon) or day.get(str(horizon)) or {}
    return found


def _summary_table(signals: list[dict], paths: dict, horizon: int) -> str:
    header = (
        "| 信号日 | 名单 | 已持有 | "
        "5 日组合 | 5 日大盘 | 5 日超额 | "
        "10 日组合 | 10 日大盘 | 10 日超额 | "
        "20 日组合 | 20 日大盘 | 20 日超额 |"
    )
    rule = "|" + "---|" * 12
    body = [header, rule]
    for signal in signals:
        picks = list(_horizon_book(signal, horizon).get("picks") or [])
        path = _frame(paths, signal["date"], horizon)
        flat = not picks
        held = 0 if path.empty else int(path["hold"].max())
        names = "空仓" if flat else "、".join(_code(symbol) for symbol in picks)
        cells = [signal["date"], names, "—" if flat else str(held)]
        for hold in HORIZONS:
            state = horizon_state(path, hold, flat=flat)
            if isinstance(state, str):
                cells.extend([state, state, state])
            else:
                cells.extend([_pct(state["port"]), _pct(state["bench"]), _pct(state["excess"])])
        body.append("| " + " | ".join(cells) + " |")
    return "\n".join(body)


def _signal_section(
    signal: dict,
    horizon: int,
    path: pd.DataFrame,
    names: dict[str, str],
    entries: dict[str, float],
) -> list[str]:
    picks = list(_horizon_book(signal, horizon).get("picks") or [])
    lines = ["", f"### {signal['date']}", ""]
    note = (signal.get("note") or "").strip()
    if note:
        lines.extend([note, ""])
    lines.extend(_board_lines(signal, horizon, names))
    if not picks:
        lines.extend(["", "这一天三个模型的前 5 名没有交集，不建仓。", ""])
        return lines
    lines.extend(["", "入场：", ""])
    lines.append("| 代码 | 名称 | 入场收盘 | 种子 0 | 种子 1 | 种子 2 |")
    lines.append("|---|---|---:|---:|---:|---:|")
    scores = _score_lookup(signal, horizon)
    for symbol in picks:
        close = entries.get(symbol)
        close_text = "—" if close is None or not np.isfinite(close) else f"{close:.2f}"
        seed_scores = scores.get(symbol, {})
        lines.append(
            f"| {_code(symbol)} | {names.get(symbol, '')} | {close_text} | "
            f"{_score(seed_scores.get(0))} | {_score(seed_scores.get(1))} | {_score(seed_scores.get(2))} |"
        )
    lines.extend(["", "累计涨跌幅：", ""])
    if path.empty:
        lines.append("信号日收盘不齐，还不能计算。")
        lines.append("")
        return lines
    headers = ["日期", "持有", *[_code(symbol) for symbol in picks], "组合", "大盘", "超额"]
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("|" + "---|" * len(headers))
    for _, row in path.iterrows():
        cells = [pd.Timestamp(row["trade_date"]).strftime("%Y-%m-%d"), str(int(row["hold"]))]
        cells.extend(_pct(row[symbol]) for symbol in picks)
        cells.extend([_pct(row["port"]), _pct(row["bench"]), _pct(row["excess"])])
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")
    return lines


def _board_lines(signal: dict, horizon: int, names: dict[str, str]) -> list[str]:
    seeds = _horizon_book(signal, horizon).get("seeds") or {}
    lines = []
    for seed in (0, 1, 2):
        board = seeds.get(str(seed)) or seeds.get(seed) or []
        if not board:
            continue
        text = "、".join(
            f"{_code(item['symbol'])} {_score(item.get('score'))} {names.get(item['symbol'], '')}".rstrip()
            for item in board
        )
        lines.append(f"种子 {seed} 前 5：{text}")
    return lines


def _score_lookup(signal: dict, horizon: int) -> dict[str, dict[int, float]]:
    found: dict[str, dict[int, float]] = {}
    seeds = _horizon_book(signal, horizon).get("seeds") or {}
    for seed in (0, 1, 2):
        for item in seeds.get(str(seed)) or seeds.get(seed) or []:
            found.setdefault(item["symbol"], {})[seed] = item.get("score")
    return found


def _positive(series: pd.Series) -> pd.Series:
    out = series.copy()
    out.index = pd.to_datetime(out.index).normalize()
    out = out.sort_index()
    out = out[~out.index.duplicated(keep="last")]
    values = out.to_numpy(dtype=np.float64)
    out = out.where(np.isfinite(values) & (values > 0))
    return out.dropna()


def _code(symbol: str) -> str:
    if symbol.endswith(".US"):
        return symbol[: -len(".US")]
    return symbol


def _pct(value) -> str:
    if value is None or not np.isfinite(value):
        return "—"
    return f"{float(value):+.2%}"


def _score(value) -> str:
    if value is None or not np.isfinite(value):
        return "—"
    return f"{float(value):+.3f}"
