from __future__ import annotations

from datetime import date, timedelta

import pandas as pd
import yfinance as yf

from .utils import normalize_ohlcv_index


def fetch_market_data(ticker: str, start_date: date, end_date: date) -> pd.DataFrame:
    """Download daily OHLCV data from Yahoo Finance."""
    ticker = ticker.strip().upper()
    if not ticker:
        raise ValueError("Ticker cannot be empty.")

    yf_end = pd.to_datetime(end_date).date() + timedelta(days=1)
    data = yf.download(
        ticker,
        start=start_date,
        end=yf_end,
        interval="1d",
        auto_adjust=False,
        progress=False,
        threads=False,
    )

    if data is None or data.empty:
        return pd.DataFrame()

    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    data = normalize_ohlcv_index(data)
    required = ["Open", "High", "Low", "Close", "Volume"]
    for column in required:
        if column not in data.columns:
            data[column] = pd.NA
    if "Adj Close" not in data.columns:
        data["Adj Close"] = data["Close"]

    return data.dropna(subset=["Close"])
