from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def add_return_significance(event_returns: pd.DataFrame) -> pd.DataFrame:
    """Add exploratory one-sample t-test p-values against zero return."""
    if event_returns.empty:
        return pd.DataFrame()

    rows = []
    for (event_type, window), group in event_returns.dropna(subset=["forward_return"]).groupby(
        ["event_type", "window"], observed=True
    ):
        returns = group["forward_return"].astype(float)
        if len(returns) < 3 or np.isclose(returns.std(ddof=1), 0):
            p_value = np.nan
            t_stat = np.nan
            note = "insufficient sample size"
        else:
            t_stat, p_value = stats.ttest_1samp(returns, popmean=0.0, nan_policy="omit")
            note = "exploratory only"
        rows.append(
            {
                "event_type": event_type,
                "window": int(window),
                "t_stat": t_stat,
                "p_value": p_value,
                "test_note": note,
            }
        )

    return pd.DataFrame(rows)


def merge_significance(stats_frame: pd.DataFrame, event_returns: pd.DataFrame) -> pd.DataFrame:
    if stats_frame.empty:
        return stats_frame
    tests = add_return_significance(event_returns)
    if tests.empty:
        return stats_frame
    return stats_frame.merge(tests, on=["event_type", "window"], how="left")
