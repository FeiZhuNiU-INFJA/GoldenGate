"""Date roles for the horizon rankers.

The report window is never used to choose the number of trees or the score sign.
A slice just before the report start is the tuning window. The final model is
refit on every row before the report start.
"""
from __future__ import annotations

import pandas as pd

RANK_HORIZONS = (5, 10, 20)
TUNE_START = "2024-01-01"
REPORT_START = "2024-06-01"


def assign_split(
    dates: pd.Series,
    tune_start: str = TUNE_START,
    report_start: str = REPORT_START,
) -> pd.DataFrame:
    """Boolean columns ``fit``, ``tune``, ``train``, and ``report``.

    ``train`` is ``fit`` plus ``tune``: every date strictly before ``report_start``.
    ``report`` starts on ``report_start``.
    """
    day = pd.to_datetime(dates).dt.normalize()
    tune = pd.Timestamp(tune_start)
    report = pd.Timestamp(report_start)
    if tune >= report:
        raise ValueError("tune_start must be earlier than report_start")
    fit = day < tune
    held = (day >= tune) & (day < report)
    return pd.DataFrame(
        {
            "fit": fit.to_numpy(),
            "tune": held.to_numpy(),
            "train": (day < report).to_numpy(),
            "report": (day >= report).to_numpy(),
        },
        index=dates.index,
    )
