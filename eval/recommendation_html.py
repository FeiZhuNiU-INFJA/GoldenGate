"""Numbers for the fixed page at docs/live/intersection.html."""
from __future__ import annotations

import json

from eval.recommendation_log import page_data


def render_data(markets: list[dict]) -> str:
    """A script the static page loads. The page itself is not rebuilt."""
    blob = json.dumps({"markets": markets}, ensure_ascii=False, separators=(",", ":"))
    blob = (
        blob.replace("<", "\\u003c")
        .replace(">", "\\u003e")
        .replace("&", "\\u0026")
        .replace("\u2028", "\\u2028")
        .replace("\u2029", "\\u2029")
    )
    return f"window.INTERSECTION = {blob};\n"


def render_html(
    signals: list[dict],
    paths: dict,
    names: dict[str, str],
    entries: dict[str, dict[str, float]],
    *,
    page_title: str = "美股交集推荐",
    kicker: str = "标普 500 · 前 5 / 第 6–15 / 其余 · 三个模型前 5 交集",
    bench: str = "标普 500",
) -> str:
    """Data script for one market. ``page_title`` is unused; the page title is fixed."""
    del page_title
    payload = page_data(signals, paths, names, entries)
    return render_data(
        [
            {
                "key": "book",
                "label": bench,
                "bench": bench,
                "kicker": kicker,
                "as_of": payload["as_of"],
                "horizons": payload["horizons"],
                "groups": payload["groups"],
            }
        ]
    )
