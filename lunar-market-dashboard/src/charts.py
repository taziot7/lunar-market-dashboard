from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from .utils import price_column


TERMINAL_TEMPLATE = "plotly_dark"
EVENT_COLORS = {
    "New Moon": "#7c5cff",
    "Full Moon": "#ffd24a",
    "Perigee": "#24d17e",
    "Apogee": "#ff6b4a",
}
METRIC_LABELS = {
    "average_return": "Average Forward Return",
    "median_return": "Median Forward Return",
    "win_rate": "Win Rate",
    "standard_deviation": "Return Volatility",
    "sharpe_like_ratio": "Sharpe-like Ratio",
}


def _base_layout(fig: go.Figure, title: str | None = None) -> go.Figure:
    fig.update_layout(
        template=TERMINAL_TEMPLATE,
        paper_bgcolor="#0d1117",
        plot_bgcolor="#10151d",
        font={"family": "Inter, -apple-system, BlinkMacSystemFont, Segoe UI, sans-serif", "color": "#d7dde7"},
        height=620,
        margin={"l": 48, "r": 32, "t": 96, "b": 72},
        hovermode="x unified",
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.06,
            "xanchor": "left",
            "x": 0,
            "bgcolor": "rgba(13,17,23,0.72)",
            "bordercolor": "rgba(255,255,255,0.08)",
            "borderwidth": 1,
        },
        title={"text": title, "x": 0.01, "xanchor": "left", "y": 0.98, "yanchor": "top"},
        xaxis={"gridcolor": "rgba(255,255,255,0.06)", "showspikes": True},
        yaxis={"gridcolor": "rgba(255,255,255,0.06)", "showspikes": True},
    )
    return fig


def plot_price_with_lunar_events(
    price_data: pd.DataFrame,
    mapped_events: pd.DataFrame,
    event_returns: pd.DataFrame,
    selected_window: int,
    chart_type: str = "Line",
) -> go.Figure:
    price_col = price_column(price_data)
    fig = go.Figure()

    if chart_type == "Candlestick" and {"Open", "High", "Low", "Close"}.issubset(price_data.columns):
        fig.add_trace(
            go.Candlestick(
                x=price_data.index,
                open=price_data["Open"],
                high=price_data["High"],
                low=price_data["Low"],
                close=price_data["Close"],
                name="Price",
                increasing_line_color="#22c55e",
                decreasing_line_color="#ef4444",
                increasing_fillcolor="#22c55e",
                decreasing_fillcolor="#ef4444",
                whiskerwidth=0.65,
            )
        )
    else:
        fig.add_trace(
            go.Scatter(
                x=price_data.index,
                y=price_data[price_col],
                mode="lines",
                name=price_col,
                line={"color": "#d2d7df", "width": 1.6},
            )
        )

    if mapped_events.empty:
        return _base_layout(fig, "Price + Lunar Overlay")

    returns_lookup = {}
    if not event_returns.empty:
        chosen = event_returns[event_returns["window"] == selected_window].copy()
        for _, row in chosen.iterrows():
            key = (pd.to_datetime(row["mapped_market_date"]).normalize(), row["event_type"])
            returns_lookup[key] = row["forward_return"]

    price_lookup = price_data[price_col].to_dict()
    high_lookup = price_data["High"].to_dict() if "High" in price_data.columns else price_lookup
    low_lookup = price_data["Low"].to_dict() if "Low" in price_data.columns else price_lookup

    for event_type in ["New Moon", "Full Moon", "Perigee", "Apogee"]:
        subset = mapped_events[mapped_events["event_type"] == event_type].dropna(subset=["mapped_market_date"])
        if subset.empty:
            continue

        x_values = []
        y_values = []
        custom = []
        for _, event in subset.iterrows():
            mapped_date = pd.to_datetime(event["mapped_market_date"]).normalize()
            if mapped_date not in price_lookup:
                continue
            px_value = float(price_lookup[mapped_date])
            if event_type == "Perigee":
                y_value = float(high_lookup.get(mapped_date, px_value)) * 1.012
                symbol = "triangle-up"
            elif event_type == "Apogee":
                y_value = float(low_lookup.get(mapped_date, px_value)) * 0.988
                symbol = "triangle-down"
            else:
                y_value = px_value
                symbol = "circle" if event_type == "Full Moon" else "diamond"

            fwd = returns_lookup.get((mapped_date, event_type), np.nan)
            distance = event.get("moon_distance_km", np.nan)
            market_context = event.get("market_context", "")
            strength_score = event.get("strength_score", np.nan)
            context_return = event.get("post_event_selected_window_return", np.nan)
            fwd_display = context_return if pd.notna(context_return) else fwd
            x_values.append(mapped_date)
            y_values.append(y_value)
            custom.append(
                [
                    event["event_timestamp_utc"],
                    event_type,
                    "" if pd.isna(distance) else f"{distance:,.0f} km",
                    px_value,
                    "" if pd.isna(fwd_display) else f"{fwd_display * 100:.2f}%",
                    mapped_date.strftime("%Y-%m-%d"),
                    market_context,
                    "" if pd.isna(strength_score) else f"{strength_score:.1f}/100",
                ]
            )

            if event_type in {"New Moon", "Full Moon"}:
                fig.add_vline(
                    x=mapped_date,
                    line_width=0.6,
                    line_dash="dash",
                    line_color=EVENT_COLORS[event_type],
                    opacity=0.18,
                )

        fig.add_trace(
            go.Scatter(
                x=x_values,
                y=y_values,
                mode="markers",
                name=event_type,
                marker={
                    "symbol": symbol,
                    "size": 12 if event_type in {"Apogee", "Perigee"} else 10,
                    "color": EVENT_COLORS[event_type],
                    "line": {"color": "rgba(255,255,255,0.95)", "width": 1.2},
                },
                customdata=custom,
                hovertemplate=(
                    "<b>%{customdata[1]}</b><br>"
                    "Mapped market date: %{customdata[5]}<br>"
                    "UTC event: %{customdata[0]}<br>"
                    "Distance: %{customdata[2]}<br>"
                    "Close: %{customdata[3]:,.2f}<br>"
                    f"Forward return +{selected_window} trading days: "
                    "%{customdata[4]}<br>"
                    "Market context: %{customdata[6]}<br>"
                    "Strength score: %{customdata[7]}<extra></extra>"
                ),
            )
        )

    fig.update_xaxes(rangeslider_visible=False)
    return _base_layout(fig, "Price + Lunar Overlay")


def plot_event_return_bars(stats_frame: pd.DataFrame, metric: str = "average_return") -> go.Figure:
    if stats_frame.empty:
        return _base_layout(go.Figure(), "Event Return Statistics")
    metric_label = METRIC_LABELS.get(metric, metric.replace("_", " ").title())
    fig = px.bar(
        stats_frame,
        x="window",
        y=metric,
        color="event_type",
        barmode="group",
        color_discrete_map=EVENT_COLORS,
        labels={"window": "Forward window", metric: metric_label},
    )
    fig.update_yaxes(tickformat=".2%")
    return _base_layout(fig, metric_label)


def plot_return_distribution(event_returns: pd.DataFrame, selected_window: int) -> go.Figure:
    if event_returns.empty:
        return _base_layout(go.Figure(), "Return Distribution")
    subset = event_returns[event_returns["window"] == selected_window]
    fig = px.box(
        subset,
        x="event_type",
        y="forward_return",
        color="event_type",
        points="outliers",
        color_discrete_map=EVENT_COLORS,
        labels={"event_type": "Event type", "forward_return": "Forward return"},
    )
    fig.update_yaxes(tickformat=".2%")
    return _base_layout(fig, f"Forward Return Distribution (+{selected_window} Trading Days)")


def plot_volatility_before_after(vol_summary: pd.DataFrame) -> go.Figure:
    if vol_summary.empty:
        return _base_layout(go.Figure(), "Volatility Before/After")
    tidy = vol_summary.melt(
        id_vars=["event_type", "window"],
        value_vars=["average_volatility_before", "average_volatility_after"],
        var_name="period",
        value_name="volatility",
    )
    tidy["period"] = tidy["period"].str.replace("average_volatility_", "", regex=False).str.title()
    fig = px.bar(
        tidy,
        x="event_type",
        y="volatility",
        color="period",
        facet_col="window",
        barmode="group",
        labels={"event_type": "Event type", "volatility": "Annualized realized volatility"},
        color_discrete_sequence=["#38bdf8", "#ffd24a"],
    )
    fig.update_yaxes(tickformat=".1%")
    return _base_layout(fig, "Average Realized Volatility Before vs After")


def plot_volatility_change_distribution(vol_events: pd.DataFrame) -> go.Figure:
    if vol_events.empty:
        return _base_layout(go.Figure(), "Volatility Change Distribution")
    fig = px.box(
        vol_events,
        x="event_type",
        y="volatility_change",
        color="event_type",
        facet_col="window",
        points="outliers",
        color_discrete_map=EVENT_COLORS,
        labels={"event_type": "Event type", "volatility_change": "After minus before volatility"},
    )
    fig.update_yaxes(tickformat=".1%")
    return _base_layout(fig, "Volatility Change Distribution")


def plot_high_low_cluster_chart(cluster_summary: pd.DataFrame) -> go.Figure:
    if cluster_summary.empty:
        return _base_layout(go.Figure(), "High/Low Clustering")
    tidy = cluster_summary.melt(
        id_vars=["event_type"],
        value_vars=["near_high_pct", "near_low_pct", "random_near_high_pct", "random_near_low_pct"],
        var_name="measure",
        value_name="rate",
    )
    labels = {
        "near_high_pct": "Events Near High",
        "near_low_pct": "Events Near Low",
        "random_near_high_pct": "Random Near High",
        "random_near_low_pct": "Random Near Low",
    }
    tidy["measure"] = tidy["measure"].map(labels)
    fig = px.bar(
        tidy,
        x="event_type",
        y="rate",
        color="measure",
        barmode="group",
        labels={"event_type": "Event type", "rate": "Share of events"},
        color_discrete_sequence=["#ffd24a", "#38bdf8", "#9b7d32", "#2f6f89"],
    )
    fig.update_yaxes(tickformat=".1%")
    return _base_layout(fig, "Event Clustering Near Local Highs/Lows")
