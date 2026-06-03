# Lunar Market Research Terminal

Lunar Market Research Terminal is a Streamlit research dashboard for overlaying lunar events on financial market charts and testing whether those events appear near price highs/lows, returns, volatility changes, or market structure.

The app analyzes four event types:

- New Moon
- Full Moon
- Lunar Apogee
- Lunar Perigee

New and full moons belong to the Moon phase cycle. Apogee and perigee belong to the Moon distance cycle. These cycles do not always align, which lets you compare phase effects, distance effects, and phase-distance combinations.

## Important Warning

This app is exploratory. It does not prove causation, predict future returns, or provide financial advice. Market results may be random, and backtests can overfit. Lunar events should not be treated as standalone trading signals.

## Features

- Fetches daily OHLCV market data with `yfinance`
- Calculates new/full moons with Skyfield moon phases
- Approximates apogee/perigee from sampled Earth-Moon distance using Skyfield and SciPy peak detection
- Maps lunar calendar events to market trading sessions
- Preserves both original UTC lunar timestamp and mapped market date
- Shows interactive Plotly line or candlestick charts
- Provides quick date presets: 3M, 6M, YTD, 1Y, 2Y, 5Y, 10Y, Max, plus a custom range
- Tests event-window forward returns after 1, 3, 5, 10, and 20 trading days
- Adds descriptive market-context ratings for each mapped event
- Compares new moon, full moon, apogee, and perigee statistics
- Studies combinations such as full moon near perigee or new moon near apogee
- Tests local high/low clustering against a random-session baseline
- Compares realized volatility before and after lunar events
- Exports CSV data and a TradingView Pine Script helper

## Project Structure

```text
lunar-market-dashboard/
  app.py
  requirements.txt
  README.md
  src/
    moon_events.py
    market_data.py
    backtest.py
    charts.py
    statistics.py
    exports.py
    utils.py
  data/
    .gitkeep
```

## Run Locally

From the project directory:

```bash
cd lunar-market-dashboard
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

Then open the local URL printed by Streamlit, usually:

```text
http://localhost:8501
```

## Streamlit Community Cloud Deployment

1. Push the `lunar-market-dashboard` folder to a GitHub repository.
2. Go to [Streamlit Community Cloud](https://streamlit.io/cloud).
3. Choose **New app**.
4. Select your GitHub repository and branch.
5. Set the main file path to:

```text
lunar-market-dashboard/app.py
```

6. Deploy.

No paid API keys are required for v1. The first run may download Skyfield's `de421.bsp` ephemeris file into `data/skyfield/`.

## Tabs

### Price + Lunar Overlay

Displays selected market data with lunar event overlays. New and full moons are shown as vertical lines. Perigee and apogee are shown as triangle markers above or below the price. Hover tooltips include event date, event type, moon distance when available, nearest mapped trading-session price, and forward return for the selected window.

The hover tooltip also shows a descriptive market context and strength score. These ratings are not predictive. They summarize how the event lined up with recent price action, moving averages, local highs/lows, and selected-window returns.

### Event Window Backtest

Calculates forward returns after each event using 1, 3, 5, 10, and 20 trading-day windows. The table includes average return, median return, win rate, best/worst return, standard deviation, Sharpe-like ratio, average max drawdown, average max upside, occurrences, and exploratory p-values.

### Combination Study

Finds phase-distance combinations:

- Full Moon near Perigee
- Full Moon near Apogee
- New Moon near Perigee
- New Moon near Apogee

You can set the near threshold to within 1, 2, 3, or 5 calendar days. The app calculates the same forward-return statistics and lists historical combination dates.

### High/Low Clustering

Tests whether lunar events occur near local market highs or lows. You can choose the local extrema lookback window and event proximity window. Results are compared with a simple random date sampling baseline.

The page explains local highs, local lows, proximity windows, and the random baseline in plain English. Higher than random may suggest clustering, similar to random suggests no meaningful clustering, and small samples are weak evidence.

### Volatility Study

Compares realized volatility before and after lunar events for 5-day and 10-day windows. The app shows average before/after volatility, volatility change, and event-level data.

### Event Data Export

Exports:

- Lunar event dataset as CSV
- Backtest results as CSV
- Combination study results as CSV
- TradingView-friendly date list as CSV
- Pine Script helper text with event timestamp arrays

The lunar event CSV includes selected-window context fields such as `market_context`, `strength_score`, `near_local_high`, `near_local_low`, 20-day pre-event return, selected-window post-event return, and 50/200-day moving-average flags.

## Data Alignment Rules

Lunar events occur on calendar dates and exact UTC timestamps. Markets only trade on sessions. The app maps event timestamps to a market session using a sidebar setting:

- `next`: map weekend or holiday events to the next available trading day
- `previous`: map weekend or holiday events to the previous available trading day

All return calculations use the mapped market date. The original lunar timestamp is preserved for auditability.

## Known Limitations

- Yahoo Finance data can be revised or unavailable for some tickers.
- Daily market data is used, so intraday event timing is not modeled.
- New/full moons are calculated with Skyfield's moon phase search.
- Apogee/perigee are approximated by sampling Earth-Moon distance every 6 hours and finding local maxima/minima. This is robust for daily market analysis but not an exact astronomical event-time calculator.
- Random baselines are simple session samples and are meant for orientation, not formal proof.
- Context ratings and strength scores are descriptive, not predictive, and should not be interpreted as trading recommendations.
- Statistical tests are exploratory and do not correct for all forms of multiple testing or strategy selection bias.

## Production Notes

- The app uses Streamlit caching for market data, lunar event calculations, backtests, clustering, and volatility studies.
- The code is split into reusable modules under `src/` so calculations can be tested or reused outside Streamlit.
- The UI is intentionally styled as a research terminal rather than an astrology-themed application.
