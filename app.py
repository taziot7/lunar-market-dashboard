from __future__ import annotations

from datetime import date, timedelta

import pandas as pd
import streamlit as st

from src.backtest import (
    calculate_all_trading_day_forward_returns,
    calculate_baseline_comparison,
    calculate_combo_events,
    calculate_event_context,
    calculate_event_stats,
    calculate_event_windows,
    calculate_high_low_clusters,
    calculate_realized_volatility,
    enrich_stats_with_baseline,
    rank_lunar_setups,
)
from src.charts import (
    plot_event_return_bars,
    plot_high_low_cluster_chart,
    plot_price_with_lunar_events,
    plot_return_distribution,
    plot_volatility_change_distribution,
    plot_volatility_before_after,
)
from src.exports import dataframe_to_csv_bytes, generate_pine_script, tradingview_date_list
from src.market_data import fetch_market_data
from src.moon_events import calculate_lunar_events
from src.statistics import merge_significance
from src.utils import (
    COMBO_THRESHOLDS,
    DEFAULT_TICKERS,
    EVENT_TYPES,
    EVENT_WINDOWS,
    LOCAL_EXTREMA_WINDOWS,
    PROXIMITY_WINDOWS,
    coerce_selected_events,
    map_events_to_trading_days,
)


DATE_PRESETS = ["3M", "6M", "YTD", "1Y", "2Y", "5Y", "10Y", "Max", "Custom"]
UI_LABELS = {
    "event_type": "Event Type",
    "window": "Window",
    "occurrences": "Number of Events",
    "total_events": "Number of Events",
    "average_return": "Average Forward Return",
    "median_return": "Median Forward Return",
    "win_rate": "Win Rate",
    "worst_return": "Worst Return",
    "best_return": "Best Return",
    "standard_deviation": "Return Volatility",
    "sharpe_like_ratio": "Sharpe-like Ratio",
    "average_max_drawdown": "Average Max Drawdown",
    "average_max_upside": "Average Max Upside",
    "event_timestamp_utc": "Original UTC Timestamp",
    "mapped_market_date": "Mapped Market Date",
    "close": "Close",
    "forward_return": "Forward Return",
    "entry_price": "Entry Price",
    "exit_price": "Exit Price",
    "market_context": "Market Context",
    "strength_score": "Strength Score",
    "near_local_high": "Near Local High",
    "near_local_low": "Near Local Low",
    "pre_event_20d_return": "Pre-Event 20D Return",
    "post_event_selected_window_return": "Selected-Window Return",
    "above_50dma": "Above 50DMA",
    "above_200dma": "Above 200DMA",
    "trend_20d_positive": "20D Trend Positive",
    "recent_volatility_percentile": "Realized Volatility Percentile",
    "volatility_regime": "Volatility Regime",
    "all_trading_days_average_return": "All Trading Days Avg Return",
    "difference_vs_all_trading_days": "Difference vs All Trading Days",
    "random_sample_average_return": "Random Sample Avg Return",
    "difference_vs_random_baseline": "Difference vs Random Baseline",
    "random_sample_std_error": "Random Sample Std Error",
    "baseline_sample_size": "Baseline Sample Size",
    "signal_quality": "Signal Quality",
    "absolute_edge": "Absolute Random Edge",
    "near_high_count": "Events Near High Count",
    "near_low_count": "Events Near Low Count",
    "near_high_pct": "Events Near High Rate",
    "near_low_pct": "Events Near Low Rate",
    "random_near_high_pct": "Random Baseline High Rate",
    "random_near_low_pct": "Random Baseline Low Rate",
    "difference_vs_random_high": "Difference vs Random Highs",
    "difference_vs_random_low": "Difference vs Random Lows",
    "average_volatility_before": "Average Volatility Before",
    "average_volatility_after": "Average Volatility After",
    "average_volatility_change": "Average Volatility Change",
    "median_volatility_change": "Median Volatility Change",
    "volatility_before": "Volatility Before",
    "volatility_after": "Volatility After",
    "volatility_change": "Volatility Change",
    "combo_partner_timestamp_utc": "Partner UTC Timestamp",
    "days_apart": "Days Apart",
    "sample_quality": "Sample Quality",
    "p_value": "p-Value",
    "t_stat": "t-Stat",
}


st.set_page_config(
    page_title="Lunar Market Research Terminal",
    page_icon=":new_moon:",
    layout="wide",
    initial_sidebar_state="expanded",
)


TERMINAL_CSS = """
<style>
    :root {
        --terminal-bg: #0d1117;
        --terminal-panel: #121821;
        --terminal-panel-soft: #161d27;
        --terminal-border: rgba(214, 181, 109, 0.18);
        --terminal-text: #d7dde7;
        --terminal-muted: #8994a5;
        --terminal-gold: #d6b56d;
        --terminal-cyan: #7eb6d8;
    }
    .stApp {
        background:
            radial-gradient(circle at top right, rgba(126, 182, 216, 0.08), transparent 26rem),
            linear-gradient(180deg, #0b0f15 0%, var(--terminal-bg) 48%, #090d12 100%);
        color: var(--terminal-text);
    }
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f151e 0%, #0b1016 100%);
        border-right: 1px solid rgba(255,255,255,0.07);
    }
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] span {
        color: var(--terminal-text);
    }
    h1, h2, h3 {
        letter-spacing: 0;
        color: #f1f4f8;
    }
    h1 {
        font-size: 2.15rem;
        font-weight: 720;
        margin-bottom: 0.25rem;
    }
    h2, h3 {
        font-weight: 680;
    }
    .terminal-subtitle {
        color: var(--terminal-muted);
        font-size: 0.98rem;
        margin-top: -0.25rem;
        margin-bottom: 1.15rem;
        max-width: 980px;
    }
    .terminal-note {
        color: var(--terminal-muted);
        border: 1px solid rgba(255,255,255,0.08);
        border-left: 3px solid var(--terminal-gold);
        background: rgba(18, 24, 33, 0.72);
        padding: 0.75rem 0.9rem;
        border-radius: 6px;
        font-size: 0.92rem;
        margin-bottom: 1rem;
    }
    [data-testid="stMetric"] {
        background: rgba(18, 24, 33, 0.72);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 8px;
        padding: 0.8rem 0.9rem;
    }
    [data-testid="stMetricLabel"] {
        color: var(--terminal-muted);
    }
    div[data-testid="stDataFrame"] {
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 8px;
        overflow: hidden;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 0.15rem;
        border-bottom: 1px solid rgba(255,255,255,0.08);
    }
    .stTabs [data-baseweb="tab"] {
        color: var(--terminal-muted);
        border-radius: 6px 6px 0 0;
        padding: 0.55rem 0.8rem;
    }
    .stTabs [aria-selected="true"] {
        color: #ffffff;
        background: rgba(214, 181, 109, 0.11);
    }
    .stDownloadButton button, .stButton button {
        border-radius: 6px;
        border: 1px solid rgba(214, 181, 109, 0.35);
        background: rgba(214, 181, 109, 0.12);
        color: #f4ead1;
        font-weight: 620;
    }
    .range-summary {
        margin: 0.35rem 0 0.9rem;
        border: 1px solid rgba(126, 182, 216, 0.25);
        background: rgba(22, 29, 39, 0.82);
        border-radius: 8px;
        padding: 0.7rem 0.75rem;
    }
    .range-summary span {
        display: block;
        color: var(--terminal-cyan);
        font-size: 0.78rem;
        font-weight: 700;
        letter-spacing: 0.04em;
        margin-bottom: 0.2rem;
    }
    .range-summary strong {
        color: #f1f4f8;
        font-size: 0.9rem;
        line-height: 1.25;
    }
</style>
"""


@st.cache_data(show_spinner=False, ttl=60 * 60)
def cached_market_data(ticker: str, start_date: date, end_date: date) -> pd.DataFrame:
    return fetch_market_data(ticker, start_date, end_date)


@st.cache_data(show_spinner=False, ttl=60 * 60 * 24)
def cached_lunar_events(start_date: date, end_date: date) -> pd.DataFrame:
    return calculate_lunar_events(start_date, end_date)


@st.cache_data(show_spinner=False)
def cached_event_windows(price_data: pd.DataFrame, mapped_events: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    return calculate_event_windows(price_data, mapped_events, EVENT_WINDOWS)


@st.cache_data(show_spinner=False)
def cached_all_day_returns(price_data: pd.DataFrame) -> pd.DataFrame:
    return calculate_all_trading_day_forward_returns(price_data, EVENT_WINDOWS)


@st.cache_data(show_spinner=False)
def cached_baseline_comparison(event_returns: pd.DataFrame, all_day_returns: pd.DataFrame) -> pd.DataFrame:
    return calculate_baseline_comparison(event_returns, all_day_returns)


@st.cache_data(show_spinner=False)
def cached_event_context(price_data: pd.DataFrame, mapped_events: pd.DataFrame, selected_window: int) -> pd.DataFrame:
    return calculate_event_context(price_data, mapped_events, selected_window)


@st.cache_data(show_spinner=False)
def cached_combo_windows(
    price_data: pd.DataFrame,
    events: pd.DataFrame,
    trading_dates: list[pd.Timestamp],
    threshold_days: int,
    mapping_direction: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    combo_events = calculate_combo_events(events, trading_dates, threshold_days, mapping_direction)
    combo_returns, combo_stats = calculate_event_windows(price_data, combo_events, EVENT_WINDOWS)
    return combo_events, combo_returns, combo_stats


@st.cache_data(show_spinner=False)
def cached_clusters(
    price_data: pd.DataFrame,
    mapped_events: pd.DataFrame,
    local_window: int,
    proximity_window: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    return calculate_high_low_clusters(price_data, mapped_events, local_window, proximity_window)


@st.cache_data(show_spinner=False)
def cached_volatility(price_data: pd.DataFrame, mapped_events: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    return calculate_realized_volatility(price_data, mapped_events)


def format_percent_columns(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    percent_columns = [
        column
        for column in frame.columns
        if any(
            token in column
            for token in ["return", "rate", "deviation", "drawdown", "upside", "volatility", "pct", "difference", "edge"]
        )
    ]
    formatted = frame.copy()
    for column in percent_columns:
        if pd.api.types.is_numeric_dtype(formatted[column]):
            formatted[column] = formatted[column].map(lambda value: "" if pd.isna(value) else f"{value * 100:.2f}%")
    return formatted


def preset_start_date(preset: str, today: date) -> date:
    if preset == "3M":
        return today - timedelta(days=92)
    if preset == "6M":
        return today - timedelta(days=183)
    if preset == "YTD":
        return date(today.year, 1, 1)
    if preset == "1Y":
        return today - timedelta(days=365)
    if preset == "2Y":
        return today - timedelta(days=365 * 2)
    if preset == "5Y":
        return today - timedelta(days=365 * 5)
    if preset == "10Y":
        return today - timedelta(days=365 * 10)
    return date(2000, 1, 1)


def style_sidebar_range(start_date: date, end_date: date, preset: str) -> None:
    st.markdown(
        f"""
        <div class="range-summary">
            <span>{preset}</span>
            <strong>{start_date:%b %d, %Y} to {end_date:%b %d, %Y}</strong>
        </div>
        """,
        unsafe_allow_html=True,
    )


def sample_size_warning(frame: pd.DataFrame, count_column: str = "occurrences") -> None:
    if frame.empty or count_column not in frame.columns:
        return
    min_count = pd.to_numeric(frame[count_column], errors="coerce").min()
    if pd.isna(min_count):
        return
    if min_count < 5:
        st.warning("Very small sample. This is not statistically meaningful.")
    elif min_count < 10:
        st.warning("Small sample size. Treat this as weak evidence.")


def ordered_columns(frame: pd.DataFrame, preferred: list[str]) -> list[str]:
    return [column for column in preferred if column in frame.columns] + [
        column for column in frame.columns if column not in preferred
    ]


def display_name(column: str) -> str:
    return UI_LABELS.get(column, column.replace("_", " ").title())


def display_table(frame: pd.DataFrame, height: int = 360) -> None:
    if frame.empty:
        st.info("Insufficient sample size for the current selection.")
        return
    st.dataframe(format_percent_columns(frame).rename(columns=UI_LABELS), width="stretch", height=height, hide_index=True)


def display_selectable_table(
    frame: pd.DataFrame,
    key: str,
    default_columns: list[str],
    essential_columns: list[str] | None = None,
    height: int = 360,
) -> None:
    if frame.empty:
        st.info("Insufficient sample size for the current selection.")
        return

    essential_columns = essential_columns or default_columns
    default_visible = [column for column in default_columns if column in frame.columns]
    essential_visible = [column for column in essential_columns if column in frame.columns]
    state_key = f"{key}_visible_columns"
    selector_key = f"{key}_selector"
    if state_key not in st.session_state:
        st.session_state[state_key] = default_visible
    if selector_key not in st.session_state:
        st.session_state[selector_key] = list(st.session_state[state_key])
    st.session_state[selector_key] = [
        column for column in st.session_state[selector_key] if column in frame.columns
    ] or default_visible

    actions = st.columns([1, 1, 1, 4])
    if actions[0].button("Show essential columns", key=f"{key}_essential"):
        st.session_state[state_key] = essential_visible
        st.session_state[selector_key] = essential_visible
    if actions[1].button("Show all columns", key=f"{key}_all"):
        st.session_state[state_key] = list(frame.columns)
        st.session_state[selector_key] = list(frame.columns)
    if actions[2].button("Reset columns", key=f"{key}_reset"):
        st.session_state[state_key] = default_visible
        st.session_state[selector_key] = default_visible

    visible = st.multiselect(
        "Visible columns",
        options=list(frame.columns),
        key=selector_key,
        format_func=display_name,
    )
    st.session_state[state_key] = visible
    if not visible:
        st.info("Choose at least one visible column.")
        return
    display_table(frame[visible], height=height)


def apply_regime_filters(frame: pd.DataFrame, filters: dict) -> pd.DataFrame:
    if frame.empty:
        return frame
    filtered = frame.copy()

    if filters["ma50"] == "Above 50DMA" and "above_50dma" in filtered.columns:
        filtered = filtered[filtered["above_50dma"] == True]
    elif filters["ma50"] == "Below 50DMA" and "above_50dma" in filtered.columns:
        filtered = filtered[filtered["above_50dma"] == False]

    if filters["ma200"] == "Above 200DMA" and "above_200dma" in filtered.columns:
        filtered = filtered[filtered["above_200dma"] == True]
    elif filters["ma200"] == "Below 200DMA" and "above_200dma" in filtered.columns:
        filtered = filtered[filtered["above_200dma"] == False]

    if filters["trend20"] == "20D trend positive" and "trend_20d_positive" in filtered.columns:
        filtered = filtered[filtered["trend_20d_positive"] == True]
    elif filters["trend20"] == "20D trend negative" and "trend_20d_positive" in filtered.columns:
        filtered = filtered[filtered["trend_20d_positive"] == False]

    if filters["volatility"] == "High volatility" and "recent_volatility_percentile" in filtered.columns:
        filtered = filtered[filtered["recent_volatility_percentile"] >= filters["vol_high_threshold"]]
    elif filters["volatility"] == "Low volatility" and "recent_volatility_percentile" in filtered.columns:
        filtered = filtered[filtered["recent_volatility_percentile"] <= filters["vol_low_threshold"]]

    return filtered


def summarize_regime_filters(filters: dict) -> str:
    active = []
    for key in ["ma50", "ma200", "trend20", "volatility"]:
        if filters[key] != "Any":
            active.append(filters[key])
    return ", ".join(active) if active else "None"


def session_context_from_all_days(all_day_returns: pd.DataFrame) -> pd.DataFrame:
    if all_day_returns.empty:
        return pd.DataFrame()
    context_cols = [
        "mapped_market_date",
        "close",
        "pre_event_20d_return",
        "trend_20d_positive",
        "above_50dma",
        "above_200dma",
        "recent_realized_volatility",
        "recent_volatility_percentile",
        "volatility_regime",
    ]
    available = [column for column in context_cols if column in all_day_returns.columns]
    return all_day_returns[available].drop_duplicates("mapped_market_date")


def analyze_combo_setups(
    price_data: pd.DataFrame,
    events: pd.DataFrame,
    all_day_returns: pd.DataFrame,
    controls: dict,
    threshold: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    combo_events = calculate_combo_events(
        events,
        list(price_data.index),
        threshold,
        controls["mapping_direction"],
    )
    if combo_events.empty:
        return combo_events, pd.DataFrame(), pd.DataFrame()

    session_context = session_context_from_all_days(all_day_returns)
    if not session_context.empty:
        combo_events = combo_events.merge(session_context, on="mapped_market_date", how="left")
    combo_events = apply_regime_filters(combo_events, controls["regime_filters"])
    if combo_events.empty:
        return combo_events, pd.DataFrame(), pd.DataFrame()

    combo_returns, combo_stats = calculate_event_windows(price_data, combo_events, EVENT_WINDOWS)
    if not combo_returns.empty and not session_context.empty:
        combo_returns = combo_returns.merge(session_context, on="mapped_market_date", how="left")
    combo_returns = apply_regime_filters(combo_returns, controls["regime_filters"])
    combo_stats = calculate_event_stats(combo_returns)
    combo_baseline = calculate_baseline_comparison(combo_returns, all_day_returns)
    combo_stats = enrich_stats_with_baseline(combo_stats, combo_baseline)
    combo_stats = merge_significance(combo_stats, combo_returns)
    return combo_events, combo_returns, combo_stats

def sidebar_controls() -> dict:
    with st.sidebar:
        st.title("Research Controls")
        st.caption("Daily data, no paid APIs. Event timestamps are UTC.")

        default_index = DEFAULT_TICKERS.index("SPY")
        selected_ticker = st.selectbox("Ticker", DEFAULT_TICKERS, index=default_index)
        custom_ticker = st.text_input(
            "Custom ticker",
            placeholder="Example: MSFT, GLD, EURUSD=X",
            help="Yahoo Finance symbols are supported. Leave blank to use the selected default.",
        )
        ticker = custom_ticker.strip().upper() or selected_ticker

        today = date.today()
        if "date_preset" not in st.session_state:
            st.session_state["date_preset"] = "10Y"
        st.caption("Date range preset")
        for row_start in range(0, 8, 2):
            preset_cols = st.columns(2)
            for offset, preset in enumerate(DATE_PRESETS[row_start : row_start + 2]):
                if preset_cols[offset].button(
                    preset,
                    key=f"preset_{preset}",
                    type="primary" if st.session_state["date_preset"] == preset else "secondary",
                ):
                    st.session_state["date_preset"] = preset
        if st.button(
            "Custom range",
            key="preset_custom",
            type="primary" if st.session_state["date_preset"] == "Custom" else "secondary",
        ):
            st.session_state["date_preset"] = "Custom"

        selected_preset = st.session_state["date_preset"]
        if selected_preset == "Custom":
            default_start = st.session_state.get("custom_start_date", today - timedelta(days=365 * 2))
            default_end = st.session_state.get("custom_end_date", today)
            date_range = st.date_input(
                "Custom date range",
                value=(default_start, default_end),
                min_value=date(1900, 1, 1),
                max_value=today,
                help="Advanced mode. Presets are easier for common lookbacks.",
            )
            if isinstance(date_range, tuple) and len(date_range) == 2:
                start_date, end_date = date_range
                st.session_state["custom_start_date"] = start_date
                st.session_state["custom_end_date"] = end_date
            else:
                start_date, end_date = default_start, default_end
        else:
            start_date = preset_start_date(selected_preset, today)
            end_date = today
        style_sidebar_range(start_date, end_date, selected_preset)

        st.divider()
        st.caption("Lunar event overlays")
        selected_events = []
        for event_type in EVENT_TYPES:
            enabled = st.checkbox(event_type, value=True)
            if enabled:
                selected_events.append(event_type)

        st.divider()
        selected_window_label = st.selectbox(
            "Event window",
            [f"±{window} trading day" if window == 1 else f"±{window} trading days" for window in EVENT_WINDOWS],
            index=1,
            help="Used for overlay hover returns and event-level distribution views.",
        )
        selected_window = int(selected_window_label.split("±")[1].split(" ")[0])
        chart_type = st.selectbox("Chart type", ["Line", "Candlestick"], index=1)
        analysis_mode = st.selectbox(
            "Analysis mode",
            [
                "Visual overlay",
                "Event-window backtest",
                "Combination study",
                "Market high/low clustering",
                "Volatility study",
            ],
        )
        mapping_direction = st.radio(
            "Non-trading-day event mapping",
            ["next", "previous"],
            index=0,
            horizontal=True,
            help="If a lunar event occurs on a weekend or market holiday, map it to the next or previous available trading session.",
        )
        st.divider()
        st.caption("Market regime filters")
        st.caption("Optional filters narrow the research set before stats and baselines are calculated.")
        ma50_filter = st.selectbox("50-day moving average", ["Any", "Above 50DMA", "Below 50DMA"])
        ma200_filter = st.selectbox("200-day moving average", ["Any", "Above 200DMA", "Below 200DMA"])
        trend20_filter = st.selectbox("20-day trend", ["Any", "20D trend positive", "20D trend negative"])
        volatility_filter = st.selectbox("Realized volatility regime", ["Any", "High volatility", "Low volatility"])
        vol_threshold = st.slider(
            "Volatility percentile threshold",
            min_value=50,
            max_value=90,
            value=70,
            step=5,
            help="High volatility means rolling 20-day realized volatility is at or above this rolling percentile. Low volatility uses the matching lower percentile.",
        )
        regime_filters = {
            "ma50": ma50_filter,
            "ma200": ma200_filter,
            "trend20": trend20_filter,
            "volatility": volatility_filter,
            "vol_high_threshold": vol_threshold / 100,
            "vol_low_threshold": (100 - vol_threshold) / 100,
        }
        st.markdown(
            f"""
            <div class="range-summary">
                <span>ACTIVE REGIME FILTERS</span>
                <strong>{summarize_regime_filters(regime_filters)}</strong>
            </div>
            """,
            unsafe_allow_html=True,
        )

    return {
        "ticker": ticker,
        "start_date": start_date,
        "end_date": end_date,
        "date_preset": selected_preset,
        "selected_events": selected_events,
        "selected_window": selected_window,
        "chart_type": chart_type,
        "analysis_mode": analysis_mode,
        "mapping_direction": mapping_direction,
        "regime_filters": regime_filters,
    }

def main() -> None:
    st.markdown(TERMINAL_CSS, unsafe_allow_html=True)
    controls = sidebar_controls()

    st.title("Lunar Market Research Terminal")
    st.markdown(
        '<div class="terminal-subtitle">Overlay lunar phase and distance-cycle events on market data, then test event-window returns, volatility, and local high/low clustering. This is an exploratory research tool, not a trading signal engine.</div>',
        unsafe_allow_html=True,
    )

    if controls["start_date"] >= controls["end_date"]:
        st.error("Choose a date range where the start date is before the end date.")
        st.stop()

    if (controls["end_date"] - controls["start_date"]).days < 90:
        st.warning("This date range is short. Statistical tables may show insufficient sample size.")

    if not controls["selected_events"]:
        st.warning("Select at least one lunar event type in the sidebar.")
        st.stop()

    with st.spinner("Loading market data and lunar events..."):
        try:
            price_data = cached_market_data(controls["ticker"], controls["start_date"], controls["end_date"])
        except Exception as exc:
            st.error(f"Yahoo Finance data request failed for {controls['ticker']}: {exc}")
            st.stop()

        if price_data.empty:
            st.warning(f"No daily OHLCV data was returned for {controls['ticker']} in the selected range.")
            st.stop()

        try:
            events = cached_lunar_events(controls["start_date"], controls["end_date"])
        except Exception as exc:
            st.error(f"Lunar event calculation failed: {exc}")
            st.stop()

    selected_events = coerce_selected_events(events, controls["selected_events"])
    mapped_events = map_events_to_trading_days(selected_events, price_data.index, controls["mapping_direction"])
    mapped_events = mapped_events.dropna(subset=["mapped_market_date"])

    if mapped_events.empty:
        st.warning("No selected lunar events could be mapped to market sessions in this date range.")
        st.stop()

    event_context = cached_event_context(price_data, mapped_events, controls["selected_window"])
    context_columns = [
        "event_type",
        "event_timestamp_utc",
        "mapped_market_date",
        "close",
        "market_context",
        "strength_score",
        "near_local_high",
        "near_local_low",
        "pre_event_20d_return",
        "trend_20d_positive",
        "post_event_selected_window_return",
        "above_50dma",
        "above_200dma",
        "distance_from_20d_high",
        "distance_from_20d_low",
        "recent_realized_volatility",
        "recent_volatility_percentile",
        "volatility_regime",
    ]
    if not event_context.empty:
        mapped_events = mapped_events.merge(
            event_context[context_columns],
            on=["event_type", "event_timestamp_utc", "mapped_market_date"],
            how="left",
        )
        mapped_events["forward_return"] = mapped_events["post_event_selected_window_return"]

    raw_mapped_count = len(mapped_events)
    active_regime_text = summarize_regime_filters(controls["regime_filters"])
    mapped_events = apply_regime_filters(mapped_events, controls["regime_filters"])
    all_day_returns = apply_regime_filters(cached_all_day_returns(price_data), controls["regime_filters"])

    if mapped_events.empty:
        st.warning("No mapped lunar events matched the active market regime filters.")
        st.stop()

    event_returns, event_stats = cached_event_windows(price_data, mapped_events)
    if not event_context.empty and not event_returns.empty:
        merge_columns = [
            column
            for column in context_columns
            if column not in {"event_type", "event_timestamp_utc", "mapped_market_date"}
        ]
        event_returns = event_returns.merge(
            event_context[["event_type", "event_timestamp_utc", "mapped_market_date", *merge_columns]],
            on=["event_type", "event_timestamp_utc", "mapped_market_date"],
            how="left",
        )
    baseline_comparison = cached_baseline_comparison(event_returns, all_day_returns)
    event_stats = enrich_stats_with_baseline(event_stats, baseline_comparison)
    event_stats = merge_significance(event_stats, event_returns)

    latest_price = float(price_data["Close"].dropna().iloc[-1])
    first_price = float(price_data["Close"].dropna().iloc[0])
    total_return = latest_price / first_price - 1.0
    filtered_mapped_count = len(mapped_events)
    unmapped_count = len(selected_events) - raw_mapped_count

    metric_cols = st.columns(5)
    metric_cols[0].metric("Ticker", controls["ticker"])
    metric_cols[1].metric("Market sessions", f"{len(price_data):,}")
    metric_cols[2].metric("Mapped events", f"{raw_mapped_count:,}", delta=f"{unmapped_count} unmapped")
    metric_cols[3].metric("Regime-filtered events", f"{filtered_mapped_count:,}", delta=f"{filtered_mapped_count - raw_mapped_count:,}")
    metric_cols[4].metric("Range return", f"{total_return * 100:.2f}%")

    st.markdown(
        f'<div class="terminal-note">Research note: lunar events are calendar events while markets trade on sessions. Original UTC timestamps and mapped market dates are preserved separately. Active regime filters: {active_regime_text}. Baseline comparison is exploratory, random sampling can vary, results can overfit, and this is not financial advice.</div>',
        unsafe_allow_html=True,
    )

    tabs = st.tabs(
        [
            "Price + Lunar Overlay",
            "Event Window Backtest",
            "Combination Study",
            "High/Low Clustering",
            "Volatility Study",
            "Best/Worst Setups",
            "Event Data Export",
        ]
    )

    with tabs[0]:
        st.subheader("Price + Lunar Overlay")
        st.caption(f"Active analysis mode: {controls['analysis_mode']}")
        st.caption("Strength ratings are descriptive, not predictive. This is exploratory research, not a trading signal.")
        fig = plot_price_with_lunar_events(
            price_data,
            mapped_events,
            event_returns,
            controls["selected_window"],
            controls["chart_type"],
        )
        st.plotly_chart(fig)
        with st.expander("Mapped lunar events", expanded=False):
            display_cols = ordered_columns(
                mapped_events,
                [
                    "event_type",
                    "event_timestamp_utc",
                    "mapped_market_date",
                    "close",
                    "forward_return",
                    "market_context",
                    "strength_score",
                    "near_local_high",
                    "near_local_low",
                    "pre_event_20d_return",
                    "post_event_selected_window_return",
                    "above_50dma",
                    "above_200dma",
                    "trend_20d_positive",
                    "recent_volatility_percentile",
                    "volatility_regime",
                    "moon_distance_km",
                    "calculation_method",
                ],
            )
            display_selectable_table(
                mapped_events[display_cols],
                key="mapped_events",
                default_columns=[
                    "event_type",
                    "event_timestamp_utc",
                    "mapped_market_date",
                    "close",
                    "forward_return",
                    "market_context",
                    "strength_score",
                    "moon_distance_km",
                ],
                essential_columns=[
                    "event_type",
                    "event_timestamp_utc",
                    "mapped_market_date",
                    "close",
                    "forward_return",
                    "market_context",
                    "strength_score",
                ],
                height=320,
            )

    with tabs[1]:
        st.subheader("Event Window Backtest")
        st.caption("Forward returns are calculated from the mapped market session to later sessions only. Backtests can overfit.")
        stats_order = [
            "event_type",
            "window",
            "occurrences",
            "average_return",
            "median_return",
            "win_rate",
            "worst_return",
            "best_return",
            "standard_deviation",
            "sharpe_like_ratio",
            "all_trading_days_average_return",
            "difference_vs_all_trading_days",
            "random_sample_average_return",
            "difference_vs_random_baseline",
            "signal_quality",
            "average_max_drawdown",
            "average_max_upside",
            "p_value",
            "sample_quality",
        ]
        event_stats_display = event_stats[ordered_columns(event_stats, stats_order)]
        sample_size_warning(event_stats, "occurrences")
        display_selectable_table(
            event_stats_display,
            key="event_stats",
            default_columns=stats_order,
            essential_columns=[
                "event_type",
                "window",
                "occurrences",
                "average_return",
                "median_return",
                "win_rate",
                "worst_return",
                "best_return",
                "standard_deviation",
                "sharpe_like_ratio",
                "difference_vs_random_baseline",
                "signal_quality",
            ],
            height=390,
        )
        st.caption("Baseline comparison is exploratory. Random sampling can vary, and results can overfit.")
        st.plotly_chart(plot_event_return_bars(event_stats, "average_return"))
        st.plotly_chart(plot_return_distribution(event_returns, controls["selected_window"]))

        with st.expander("Event-level forward returns", expanded=False):
            event_return_order = [
                "event_type",
                "event_timestamp_utc",
                "mapped_market_date",
                "window",
                "close",
                "forward_return",
                "market_context",
                "strength_score",
                "near_local_high",
                "near_local_low",
                "pre_event_20d_return",
                "post_event_selected_window_return",
                "above_50dma",
                "above_200dma",
                "trend_20d_positive",
                "recent_volatility_percentile",
                "volatility_regime",
                "moon_distance_km",
            ]
            display_selectable_table(
                event_returns[ordered_columns(event_returns, event_return_order)],
                key="event_returns",
                default_columns=event_return_order,
                essential_columns=[
                    "event_type",
                    "event_timestamp_utc",
                    "mapped_market_date",
                    "close",
                    "forward_return",
                    "market_context",
                    "strength_score",
                ],
                height=360,
            )

    with tabs[2]:
        st.subheader("Combination Study")
        threshold = st.select_slider(
            "Near threshold",
            options=COMBO_THRESHOLDS,
            value=3,
            format_func=lambda value: f"within {value} day" if value == 1 else f"within {value} days",
            help="A phase event is paired with its nearest apogee/perigee if the distance event falls within this calendar-day threshold.",
        )
        combo_events, combo_returns, combo_stats = analyze_combo_setups(
            price_data,
            events,
            all_day_returns,
            controls,
            threshold,
        )
        if combo_events.empty:
            st.info("Insufficient sample size: no phase-distance combinations matched this threshold and regime filter.")
        else:
            sample_size_warning(combo_stats, "occurrences")
            combo_stats_order = [
                "event_type",
                "window",
                "occurrences",
                "average_return",
                "median_return",
                "win_rate",
                "worst_return",
                "best_return",
                "standard_deviation",
                "sharpe_like_ratio",
                "difference_vs_random_baseline",
                "difference_vs_all_trading_days",
                "signal_quality",
                "p_value",
                "sample_quality",
            ]
            display_selectable_table(
                combo_stats[ordered_columns(combo_stats, combo_stats_order)],
                key="combo_stats",
                default_columns=combo_stats_order,
                essential_columns=[
                    "event_type",
                    "window",
                    "occurrences",
                    "average_return",
                    "median_return",
                    "win_rate",
                    "worst_return",
                    "best_return",
                ],
                height=330,
            )
            st.plotly_chart(plot_event_return_bars(combo_stats, "average_return"))
            combo_cols = [
                "event_type",
                "event_timestamp_utc",
                "mapped_market_date",
                "combo_partner_timestamp_utc",
                "days_apart",
                "moon_distance_km",
            ]
            st.markdown("Historical combo dates")
            display_selectable_table(
                combo_events[combo_cols],
                key="combo_events",
                default_columns=combo_cols,
                essential_columns=["event_type", "event_timestamp_utc", "mapped_market_date", "days_apart"],
                height=330,
            )

    with tabs[3]:
        st.subheader("High/Low Clustering")
        st.markdown(
            "This test checks whether lunar events occur unusually often near recent local highs or lows compared with random dates."
        )
        definition_cols = st.columns(4)
        definition_cols[0].info(f"Local high: highest close within a centered lookback window, currently {LOCAL_EXTREMA_WINDOWS[2]} trading days by default.")
        definition_cols[1].info("Local low: lowest close within the same centered lookback window.")
        definition_cols[2].info("Event proximity window: how many trading sessions around a high or low count as nearby.")
        definition_cols[3].info("Random baseline: random market sessions sampled with the same event count.")
        cluster_cols = st.columns(2)
        local_window = cluster_cols[0].selectbox(
            "Local high/low lookback window",
            LOCAL_EXTREMA_WINDOWS,
            index=2,
            help="A centered rolling window is used to identify recent local highs and lows for exploratory clustering.",
        )
        proximity_window = cluster_cols[1].selectbox(
            "Event proximity window",
            PROXIMITY_WINDOWS,
            index=1,
            format_func=lambda value: f"±{value} trading day" if value == 1 else f"±{value} trading days",
        )
        cluster_summary, cluster_events = cached_clusters(price_data, mapped_events, local_window, proximity_window)
        if not cluster_summary.empty:
            total_events = cluster_summary["total_events"].sum()
            near_high_pct = (
                cluster_summary["near_high_count"].sum() / total_events if total_events else 0
            )
            near_low_pct = (
                cluster_summary["near_low_count"].sum() / total_events if total_events else 0
            )
            random_high_pct = (
                (cluster_summary["random_near_high_pct"] * cluster_summary["total_events"]).sum() / total_events
                if total_events
                else 0
            )
            random_low_pct = (
                (cluster_summary["random_near_low_pct"] * cluster_summary["total_events"]).sum() / total_events
                if total_events
                else 0
            )
            summary_cards = st.columns(5)
            summary_cards[0].metric("Events near highs", f"{near_high_pct * 100:.1f}%")
            summary_cards[1].metric("Events near lows", f"{near_low_pct * 100:.1f}%")
            summary_cards[2].metric("Random near highs", f"{random_high_pct * 100:.1f}%")
            summary_cards[3].metric("Random near lows", f"{random_low_pct * 100:.1f}%")
            summary_cards[4].metric(
                "Difference vs random",
                f"{((near_high_pct + near_low_pct) - (random_high_pct + random_low_pct)) * 100:.1f} pp",
            )

        sample_size_warning(cluster_summary, "total_events")
        st.caption(
            "Higher than random may suggest clustering. Similar to random suggests no meaningful clustering. Small samples are weak evidence."
        )
        cluster_order = [
            "event_type",
            "total_events",
            "near_high_count",
            "near_low_count",
            "near_high_pct",
            "near_low_pct",
            "random_near_high_pct",
            "random_near_low_pct",
            "difference_vs_random_high",
            "difference_vs_random_low",
            "sample_quality",
        ]
        display_selectable_table(
            cluster_summary[ordered_columns(cluster_summary, cluster_order)],
            key="cluster_summary",
            default_columns=cluster_order,
            essential_columns=[
                "event_type",
                "total_events",
                "near_high_pct",
                "near_low_pct",
                "random_near_high_pct",
                "random_near_low_pct",
                "difference_vs_random_high",
                "difference_vs_random_low",
            ],
            height=290,
        )
        st.plotly_chart(plot_high_low_cluster_chart(cluster_summary))
        st.markdown("Event-level clustering list")
        cluster_event_order = [
            "event_type",
            "event_timestamp_utc",
            "mapped_market_date",
            "near_local_high",
            "near_local_low",
        ]
        display_selectable_table(
            cluster_events[ordered_columns(cluster_events, cluster_event_order)],
            key="cluster_events",
            default_columns=cluster_event_order,
            essential_columns=cluster_event_order,
            height=320,
        )

    with tabs[4]:
        st.subheader("Volatility Study")
        vol_summary, vol_events = cached_volatility(price_data, mapped_events)
        sample_size_warning(vol_summary, "occurrences")
        vol_summary_order = [
            "event_type",
            "window",
            "occurrences",
            "average_volatility_before",
            "average_volatility_after",
            "average_volatility_change",
            "median_volatility_change",
            "sample_quality",
        ]
        display_selectable_table(
            vol_summary[ordered_columns(vol_summary, vol_summary_order)],
            key="vol_summary",
            default_columns=vol_summary_order,
            essential_columns=[
                "event_type",
                "window",
                "occurrences",
                "average_volatility_before",
                "average_volatility_after",
                "average_volatility_change",
            ],
            height=300,
        )
        st.plotly_chart(plot_volatility_before_after(vol_summary))
        if not vol_events.empty:
            st.plotly_chart(plot_volatility_change_distribution(vol_events))
        st.markdown("Event-level volatility")
        vol_event_order = [
            "event_type",
            "event_timestamp_utc",
            "mapped_market_date",
            "window",
            "volatility_before",
            "volatility_after",
            "volatility_change",
        ]
        display_selectable_table(
            vol_events[ordered_columns(vol_events, vol_event_order)],
            key="vol_events",
            default_columns=vol_event_order,
            essential_columns=vol_event_order,
            height=340,
        )

    with tabs[5]:
        st.subheader("Best/Worst Lunar Setups")
        st.caption(
            "Ranks event types and phase-distance combinations by historical average return, win rate, sample size, return volatility, and difference versus random baseline. Signal Quality is descriptive, not predictive."
        )
        setup_threshold = st.select_slider(
            "Combination threshold for setup ranking",
            options=COMBO_THRESHOLDS,
            value=3,
            format_func=lambda value: f"within {value} day" if value == 1 else f"within {value} days",
        )
        setup_combo_events, setup_combo_returns, setup_combo_stats = analyze_combo_setups(
            price_data,
            events,
            all_day_returns,
            controls,
            setup_threshold,
        )
        setup_rankings = rank_lunar_setups([event_stats, setup_combo_stats])
        if setup_rankings.empty:
            st.info("No setup rankings are available for the current filters.")
        else:
            ranking_order = [
                "event_type",
                "window",
                "occurrences",
                "average_return",
                "win_rate",
                "standard_deviation",
                "difference_vs_random_baseline",
                "difference_vs_all_trading_days",
                "signal_quality",
                "absolute_edge",
            ]
            display_selectable_table(
                setup_rankings[ordered_columns(setup_rankings, ranking_order)],
                key="setup_rankings",
                default_columns=ranking_order,
                essential_columns=[
                    "event_type",
                    "window",
                    "occurrences",
                    "average_return",
                    "win_rate",
                    "standard_deviation",
                    "difference_vs_random_baseline",
                    "signal_quality",
                ],
                height=430,
            )
            st.caption(
                "Strong or moderate ratings require enough observations plus return, win-rate, and baseline-difference alignment. Weak ratings often mean the sample is too small or the edge is unstable."
            )

    with tabs[6]:
        st.subheader("Event Data Export")
        st.caption("Lunar events CSV includes the selected-window market context and strength-score columns.")
        tv_dates = tradingview_date_list(mapped_events)
        combo_events_export, combo_returns_export, combo_stats_export = analyze_combo_setups(
            price_data,
            events,
            all_day_returns,
            controls,
            3,
        )

        export_cols = st.columns(4)
        export_cols[0].download_button(
            "Lunar events CSV",
            data=dataframe_to_csv_bytes(mapped_events),
            file_name=f"{controls['ticker']}_lunar_events.csv",
            mime="text/csv",
        )
        export_cols[1].download_button(
            "Backtest CSV",
            data=dataframe_to_csv_bytes(event_stats),
            file_name=f"{controls['ticker']}_lunar_backtest_stats.csv",
            mime="text/csv",
        )
        export_cols[2].download_button(
            "Combination CSV",
            data=dataframe_to_csv_bytes(combo_stats_export),
            file_name=f"{controls['ticker']}_lunar_combo_stats.csv",
            mime="text/csv",
            disabled=combo_stats_export.empty,
        )
        export_cols[3].download_button(
            "TradingView dates CSV",
            data=dataframe_to_csv_bytes(tv_dates),
            file_name=f"{controls['ticker']}_tradingview_lunar_dates.csv",
            mime="text/csv",
        )

        pine_script = generate_pine_script(mapped_events, f"{controls['ticker']} Lunar Event Dates")
        st.download_button(
            "Download Pine Script helper",
            data=pine_script.encode("utf-8"),
            file_name=f"{controls['ticker']}_lunar_events.pine",
            mime="text/plain",
        )
        st.code(pine_script, language="pine")

        with st.expander("TradingView-friendly date list", expanded=False):
            display_selectable_table(
                tv_dates,
                key="tv_dates",
                default_columns=list(tv_dates.columns),
                essential_columns=["event_type", "event_date", "mapped_market_date", "timestamp_utc"],
                height=300,
            )

    st.caption(
        "Exploratory research only. Correlation does not imply causation. Backtests may overfit and lunar events should not be treated as standalone trading signals."
    )


if __name__ == "__main__":
    main()