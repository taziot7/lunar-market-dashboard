from __future__ import annotations

from datetime import date, datetime, timezone
from typing import Iterable

import numpy as np
import pandas as pd


DEFAULT_TICKERS = ["SPY", "QQQ", "DIA", "IWM", "BTC-USD", "ETH-USD", "AAPL", "TSLA", "NVDA"]
EVENT_TYPES = ["New Moon", "Full Moon", "Apogee", "Perigee"]
EVENT_WINDOWS = [1, 3, 5, 10, 20]
COMBO_THRESHOLDS = [1, 2, 3, 5]
LOCAL_EXTREMA_WINDOWS = [5, 10, 20, 50]
PROXIMITY_WINDOWS = [1, 3, 5]


def normalize_date(value: date | datetime | str | pd.Timestamp) -> pd.Timestamp:
    """Return a timezone-naive midnight timestamp for date comparisons."""
    ts = pd.to_datetime(value)
    if ts.tzinfo is not None:
        ts = ts.tz_convert("UTC").tz_localize(None)
    return ts.normalize()


def normalize_ohlcv_index(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    out = frame.copy()
    out.index = pd.to_datetime(out.index)
    if out.index.tz is not None:
        out.index = out.index.tz_convert("UTC").tz_localize(None)
    out.index = out.index.normalize()
    return out.sort_index()


def to_utc_timestamp(value: date | datetime | str | pd.Timestamp) -> pd.Timestamp:
    ts = pd.to_datetime(value)
    if ts.tzinfo is None:
        ts = ts.tz_localize(timezone.utc)
    else:
        ts = ts.tz_convert("UTC")
    return ts


def map_events_to_trading_days(
    events: pd.DataFrame,
    trading_index: Iterable[pd.Timestamp],
    direction: str = "next",
) -> pd.DataFrame:
    """Map calendar lunar timestamps to tradable market sessions.

    The original lunar timestamp and calendar event date are preserved. If an
    event lands on a weekend or market holiday, `direction="next"` maps it to
    the next available session, while `direction="previous"` maps it backward.
    """
    if events.empty:
        return events.copy()

    if direction not in {"next", "previous"}:
        raise ValueError("direction must be 'next' or 'previous'")

    sessions = pd.DatetimeIndex(pd.to_datetime(list(trading_index))).sort_values().normalize()
    if len(sessions) == 0:
        out = events.copy()
        out["mapped_market_date"] = pd.NaT
        out["mapped_session_found"] = False
        out["mapping_direction"] = direction
        return out

    mapped_dates: list[pd.Timestamp | pd.NaT] = []
    found: list[bool] = []

    for event_ts in pd.to_datetime(events["event_timestamp_utc"], utc=True):
        event_date = event_ts.tz_convert("UTC").tz_localize(None).normalize()
        if direction == "next":
            pos = sessions.searchsorted(event_date, side="left")
            if pos >= len(sessions):
                mapped_dates.append(pd.NaT)
                found.append(False)
            else:
                mapped_dates.append(sessions[pos])
                found.append(True)
        else:
            pos = sessions.searchsorted(event_date, side="right") - 1
            if pos < 0:
                mapped_dates.append(pd.NaT)
                found.append(False)
            else:
                mapped_dates.append(sessions[pos])
                found.append(True)

    out = events.copy()
    out["event_date"] = pd.to_datetime(out["event_timestamp_utc"], utc=True).dt.date
    out["mapped_market_date"] = pd.to_datetime(mapped_dates)
    out["mapped_session_found"] = found
    out["mapping_direction"] = direction
    return out


def price_column(price_data: pd.DataFrame) -> str:
    for column in ["Adj Close", "Close"]:
        if column in price_data.columns:
            return column
    raise KeyError("Price data must contain 'Adj Close' or 'Close'.")


def format_percent(value: float | int | None) -> str:
    if value is None or pd.isna(value):
        return "n/a"
    return f"{value * 100:.2f}%"


def safe_ratio(numerator: float, denominator: float) -> float:
    if denominator is None or np.isnan(denominator) or denominator == 0:
        return np.nan
    return numerator / denominator


def coerce_selected_events(events: pd.DataFrame, selected_event_types: list[str]) -> pd.DataFrame:
    if events.empty:
        return events.copy()
    return events[events["event_type"].isin(selected_event_types)].copy()
