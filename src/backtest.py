from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

from .utils import EVENT_WINDOWS, map_events_to_trading_days, price_column, safe_ratio


def calculate_forward_returns(
    price_data: pd.DataFrame,
    mapped_events: pd.DataFrame,
    windows: Iterable[int] = EVENT_WINDOWS,
) -> pd.DataFrame:
    """Calculate event-level forward returns from mapped market sessions."""
    if price_data.empty or mapped_events.empty:
        return pd.DataFrame()

    prices = price_data[price_column(price_data)].dropna()
    sessions = pd.DatetimeIndex(prices.index).normalize()
    session_to_pos = {session: pos for pos, session in enumerate(sessions)}
    rows = []

    for _, event in mapped_events.dropna(subset=["mapped_market_date"]).iterrows():
        mapped_date = pd.to_datetime(event["mapped_market_date"]).normalize()
        if mapped_date not in session_to_pos:
            continue

        pos = session_to_pos[mapped_date]
        entry_price = float(prices.iloc[pos])

        for window in windows:
            exit_pos = pos + int(window)
            if exit_pos >= len(prices):
                continue
            exit_price = float(prices.iloc[exit_pos])
            window_prices = prices.iloc[pos : exit_pos + 1].astype(float)
            forward_return = (exit_price / entry_price) - 1.0
            max_drawdown = (window_prices.min() / entry_price) - 1.0
            max_upside = (window_prices.max() / entry_price) - 1.0

            rows.append(
                {
                    "event_timestamp_utc": event["event_timestamp_utc"],
                    "event_date": event.get("event_date"),
                    "mapped_market_date": mapped_date,
                    "event_type": event["event_type"],
                    "window": int(window),
                    "entry_price": entry_price,
                    "exit_price": exit_price,
                    "forward_return": forward_return,
                    "max_drawdown": max_drawdown,
                    "max_upside": max_upside,
                    "moon_distance_km": event.get("moon_distance_km", np.nan),
                    "calculation_method": event.get("calculation_method", ""),
                }
            )

    return pd.DataFrame(rows)


def calculate_event_stats(event_returns: pd.DataFrame) -> pd.DataFrame:
    """Aggregate event-window return statistics by event type and window."""
    if event_returns.empty:
        return pd.DataFrame()

    rows = []
    grouped = event_returns.dropna(subset=["forward_return"]).groupby(["event_type", "window"], observed=True)
    for (event_type, window), group in grouped:
        returns = group["forward_return"].astype(float)
        std = returns.std(ddof=1)
        avg_return = returns.mean()
        rows.append(
            {
                "event_type": event_type,
                "window": int(window),
                "average_return": avg_return,
                "median_return": returns.median(),
                "win_rate": (returns > 0).mean(),
                "best_return": returns.max(),
                "worst_return": returns.min(),
                "standard_deviation": std,
                "sharpe_like_ratio": safe_ratio(avg_return, std),
                "average_max_drawdown": group["max_drawdown"].mean(),
                "average_max_upside": group["max_upside"].mean(),
                "occurrences": int(returns.count()),
                "sample_quality": "ok" if returns.count() >= 10 else "insufficient sample size",
            }
        )
    return pd.DataFrame(rows).sort_values(["event_type", "window"]).reset_index(drop=True)


def calculate_event_windows(
    price_data: pd.DataFrame,
    mapped_events: pd.DataFrame,
    windows: Iterable[int] = EVENT_WINDOWS,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    event_returns = calculate_forward_returns(price_data, mapped_events, windows)
    return event_returns, calculate_event_stats(event_returns)


def calculate_event_context(
    price_data: pd.DataFrame,
    mapped_events: pd.DataFrame,
    selected_window: int = 5,
) -> pd.DataFrame:
    """Score each mapped event's market context with transparent descriptive rules."""
    if price_data.empty or mapped_events.empty:
        return pd.DataFrame()

    prices = price_data[price_column(price_data)].dropna().astype(float)
    sessions = pd.DatetimeIndex(prices.index).normalize()
    session_to_pos = {session: pos for pos, session in enumerate(sessions)}
    ma50 = prices.rolling(50, min_periods=50).mean()
    ma200 = prices.rolling(200, min_periods=200).mean()
    log_returns = np.log(prices / prices.shift(1))
    trailing_volatility = log_returns.rolling(20, min_periods=10).std(ddof=1) * np.sqrt(252)
    volatility_percentile = trailing_volatility.rolling(252, min_periods=60).apply(
        lambda values: pd.Series(values).rank(pct=True).iloc[-1],
        raw=False,
    )
    rows = []

    for _, event in mapped_events.dropna(subset=["mapped_market_date"]).iterrows():
        mapped_date = pd.to_datetime(event["mapped_market_date"]).normalize()
        if mapped_date not in session_to_pos:
            continue

        pos = session_to_pos[mapped_date]
        close = float(prices.iloc[pos])
        has_context = pos >= 20 and pos + selected_window < len(prices)

        pre_return = np.nan
        post_return = np.nan
        near_high = False
        near_low = False
        distance_from_high = np.nan
        distance_from_low = np.nan
        recent_volatility = float(trailing_volatility.iloc[pos]) if pd.notna(trailing_volatility.iloc[pos]) else np.nan
        recent_volatility_percentile = (
            float(volatility_percentile.iloc[pos]) if pd.notna(volatility_percentile.iloc[pos]) else np.nan
        )
        market_context = "Insufficient Data"
        strength_score = np.nan

        if has_context:
            pre_price = float(prices.iloc[pos - 20])
            future_price = float(prices.iloc[pos + selected_window])
            pre_return = close / pre_price - 1.0
            post_return = future_price / close - 1.0

            trailing_window = prices.iloc[pos - 19 : pos + 1]
            trailing_high = float(trailing_window.max())
            trailing_low = float(trailing_window.min())
            distance_from_high = close / trailing_high - 1.0
            distance_from_low = close / trailing_low - 1.0
            near_high = distance_from_high >= -0.03
            near_low = distance_from_low <= 0.03

            daily_vol = recent_volatility / np.sqrt(252) if pd.notna(recent_volatility) and recent_volatility > 0 else 0.01
            expected_window_move = max(daily_vol * np.sqrt(selected_window), 0.005)
            small_forward_move = abs(post_return) <= max(0.01, expected_window_move * 0.5)
            range_bound = abs(pre_return) <= 0.03
            same_direction = np.sign(pre_return) == np.sign(post_return) and abs(pre_return) > 0.01 and abs(post_return) > 0.005

            if near_high and post_return < 0:
                market_context = "Possible Local Top"
            elif near_low and post_return > 0:
                market_context = "Possible Local Bottom"
            elif range_bound and small_forward_move:
                market_context = "Chop / Neutral"
            elif same_direction:
                market_context = "Trend Continuation"
            else:
                market_context = "Chop / Neutral"

            high_proximity_score = max(0.0, 1.0 - abs(distance_from_high) / 0.03) if near_high else 0.0
            low_proximity_score = max(0.0, 1.0 - abs(distance_from_low) / 0.03) if near_low else 0.0
            proximity_score = max(high_proximity_score, low_proximity_score) * 30
            move_score = min(abs(post_return) / (expected_window_move * 1.5), 1.0) * 30
            volatility_adjusted_score = min(abs(post_return) / expected_window_move, 1.0) * 20
            trend_score = 15 if same_direction else 8 if market_context in {"Possible Local Top", "Possible Local Bottom"} else 4
            context_bonus = 5 if market_context in {"Possible Local Top", "Possible Local Bottom", "Trend Continuation"} else 0
            strength_score = round(min(100.0, proximity_score + move_score + volatility_adjusted_score + trend_score + context_bonus), 1)

        above_50dma = bool(close > ma50.iloc[pos]) if pd.notna(ma50.iloc[pos]) else pd.NA
        above_200dma = bool(close > ma200.iloc[pos]) if pd.notna(ma200.iloc[pos]) else pd.NA

        rows.append(
            {
                "event_type": event["event_type"],
                "event_timestamp_utc": event["event_timestamp_utc"],
                "mapped_market_date": mapped_date,
                "close": close,
                "market_context": market_context,
                "strength_score": strength_score,
                "near_local_high": near_high,
                "near_local_low": near_low,
                "pre_event_20d_return": pre_return,
                "trend_20d_positive": bool(pre_return > 0) if pd.notna(pre_return) else pd.NA,
                "post_event_selected_window_return": post_return,
                "above_50dma": above_50dma,
                "above_200dma": above_200dma,
                "distance_from_20d_high": distance_from_high,
                "distance_from_20d_low": distance_from_low,
                "recent_realized_volatility": recent_volatility,
                "recent_volatility_percentile": recent_volatility_percentile,
                "volatility_regime": "High"
                if pd.notna(recent_volatility_percentile) and recent_volatility_percentile >= 0.7
                else "Low"
                if pd.notna(recent_volatility_percentile) and recent_volatility_percentile <= 0.3
                else "Middle",
            }
        )

    return pd.DataFrame(rows)


def calculate_session_context(price_data: pd.DataFrame) -> pd.DataFrame:
    """Calculate market-regime context for every trading session."""
    if price_data.empty:
        return pd.DataFrame()

    prices = price_data[price_column(price_data)].dropna().astype(float)
    ma50 = prices.rolling(50, min_periods=50).mean()
    ma200 = prices.rolling(200, min_periods=200).mean()
    log_returns = np.log(prices / prices.shift(1))
    trailing_volatility = log_returns.rolling(20, min_periods=10).std(ddof=1) * np.sqrt(252)
    volatility_percentile = trailing_volatility.rolling(252, min_periods=60).apply(
        lambda values: pd.Series(values).rank(pct=True).iloc[-1],
        raw=False,
    )

    context = pd.DataFrame(
        {
            "mapped_market_date": pd.DatetimeIndex(prices.index).normalize(),
            "close": prices.to_numpy(),
            "pre_event_20d_return": prices / prices.shift(20) - 1.0,
            "above_50dma": prices > ma50,
            "above_200dma": prices > ma200,
            "recent_realized_volatility": trailing_volatility,
            "recent_volatility_percentile": volatility_percentile,
        },
        index=prices.index,
    )
    context["trend_20d_positive"] = (context["pre_event_20d_return"] > 0).astype("object")
    context.loc[context["pre_event_20d_return"].isna(), "trend_20d_positive"] = pd.NA
    context["volatility_regime"] = np.select(
        [
            context["recent_volatility_percentile"] >= 0.7,
            context["recent_volatility_percentile"] <= 0.3,
        ],
        ["High", "Low"],
        default="Middle",
    )
    context.loc[context["recent_volatility_percentile"].isna(), "volatility_regime"] = pd.NA
    return context.reset_index(drop=True)


def calculate_all_trading_day_forward_returns(
    price_data: pd.DataFrame,
    windows: Iterable[int] = EVENT_WINDOWS,
) -> pd.DataFrame:
    """Forward returns for every eligible trading day, used as a baseline."""
    if price_data.empty:
        return pd.DataFrame()

    prices = price_data[price_column(price_data)].dropna().astype(float)
    session_context = calculate_session_context(price_data).set_index("mapped_market_date")
    rows = []
    for pos, mapped_date in enumerate(pd.DatetimeIndex(prices.index).normalize()):
        entry_price = float(prices.iloc[pos])
        for window in windows:
            exit_pos = pos + int(window)
            if exit_pos >= len(prices):
                continue
            row = {
                "mapped_market_date": mapped_date,
                "window": int(window),
                "entry_price": entry_price,
                "exit_price": float(prices.iloc[exit_pos]),
                "forward_return": float(prices.iloc[exit_pos] / entry_price - 1.0),
            }
            if mapped_date in session_context.index:
                row.update(session_context.loc[mapped_date].to_dict())
            rows.append(row)
    return pd.DataFrame(rows)


def rate_signal_quality(
    occurrences: int,
    average_return: float,
    win_rate: float,
    difference_vs_random_baseline: float,
) -> str:
    """Transparent descriptive rating, not a predictive signal."""
    if occurrences < 10 or pd.isna(average_return) or pd.isna(win_rate) or pd.isna(difference_vs_random_baseline):
        return "Weak / insufficient evidence"

    directionally_aligned = (average_return > 0 and win_rate >= 0.55) or (average_return < 0 and win_rate <= 0.45)
    no_visible_edge = abs(difference_vs_random_baseline) < 0.0025 and 0.45 <= win_rate <= 0.55
    if no_visible_edge:
        return "No visible edge"
    if occurrences >= 30 and abs(difference_vs_random_baseline) >= 0.015 and directionally_aligned:
        return "Strong historical pattern"
    if occurrences >= 15 and abs(difference_vs_random_baseline) >= 0.0075 and abs(win_rate - 0.5) >= 0.03:
        return "Moderate historical pattern"
    return "Weak / insufficient evidence"


def calculate_baseline_comparison(
    event_returns: pd.DataFrame,
    all_day_returns: pd.DataFrame,
    random_trials: int = 500,
    random_seed: int = 7,
) -> pd.DataFrame:
    """Compare event returns with all-day and random-sample baselines."""
    if event_returns.empty or all_day_returns.empty:
        return pd.DataFrame()

    rng = np.random.default_rng(random_seed)
    rows = []
    for (event_type, window), group in event_returns.dropna(subset=["forward_return"]).groupby(
        ["event_type", "window"], observed=True
    ):
        returns = group["forward_return"].astype(float)
        baseline = all_day_returns[all_day_returns["window"] == window]["forward_return"].dropna().astype(float)
        if returns.empty or baseline.empty:
            continue

        n = int(returns.count())
        average_return = float(returns.mean())
        win_rate = float((returns > 0).mean())
        all_days_average = float(baseline.mean())
        random_means = []
        for _ in range(random_trials):
            sampled = rng.choice(baseline.to_numpy(), size=n, replace=n > len(baseline))
            random_means.append(float(np.mean(sampled)))
        random_average = float(np.mean(random_means))
        difference_vs_random = average_return - random_average

        rows.append(
            {
                "event_type": event_type,
                "window": int(window),
                "all_trading_days_average_return": all_days_average,
                "difference_vs_all_trading_days": average_return - all_days_average,
                "random_sample_average_return": random_average,
                "difference_vs_random_baseline": difference_vs_random,
                "random_sample_std_error": float(np.std(random_means, ddof=1)) if len(random_means) > 1 else np.nan,
                "baseline_sample_size": int(baseline.count()),
                "signal_quality": rate_signal_quality(n, average_return, win_rate, difference_vs_random),
            }
        )

    return pd.DataFrame(rows).sort_values(["event_type", "window"]).reset_index(drop=True)


def enrich_stats_with_baseline(event_stats: pd.DataFrame, baseline_comparison: pd.DataFrame) -> pd.DataFrame:
    if event_stats.empty or baseline_comparison.empty:
        return event_stats
    return event_stats.merge(baseline_comparison, on=["event_type", "window"], how="left")


def rank_lunar_setups(stats_frames: list[pd.DataFrame]) -> pd.DataFrame:
    """Rank ordinary event and combination setups by descriptive historical metrics."""
    usable = [frame.copy() for frame in stats_frames if frame is not None and not frame.empty]
    if not usable:
        return pd.DataFrame()
    combined = pd.concat(usable, ignore_index=True, sort=False)
    required = {"event_type", "window", "average_return", "win_rate", "occurrences", "standard_deviation"}
    if not required.issubset(combined.columns):
        return pd.DataFrame()
    if "difference_vs_random_baseline" not in combined.columns:
        combined["difference_vs_random_baseline"] = np.nan
    if "difference_vs_all_trading_days" not in combined.columns:
        combined["difference_vs_all_trading_days"] = np.nan
    if "signal_quality" not in combined.columns:
        combined["signal_quality"] = combined.apply(
            lambda row: rate_signal_quality(
                int(row["occurrences"]),
                row["average_return"],
                row["win_rate"],
                row.get("difference_vs_random_baseline", np.nan),
            ),
            axis=1,
        )
    ranked = combined[
        [
            "event_type",
            "window",
            "occurrences",
            "average_return",
            "win_rate",
            "standard_deviation",
            "difference_vs_random_baseline",
            "difference_vs_all_trading_days",
            "signal_quality",
        ]
    ].copy()
    ranked["absolute_edge"] = ranked["difference_vs_random_baseline"].abs()
    return ranked.sort_values(
        ["absolute_edge", "average_return", "win_rate", "occurrences"],
        ascending=[False, False, False, False],
    ).reset_index(drop=True)


def calculate_combo_events(
    events: pd.DataFrame,
    trading_index: Iterable[pd.Timestamp],
    threshold_days: int = 3,
    mapping_direction: str = "next",
) -> pd.DataFrame:
    """Create phase-distance combination signals within a calendar-day threshold."""
    if events.empty:
        return pd.DataFrame()

    phase_events = events[events["event_type"].isin(["New Moon", "Full Moon"])].copy()
    distance_events = events[events["event_type"].isin(["Apogee", "Perigee"])].copy()
    if phase_events.empty or distance_events.empty:
        return pd.DataFrame()

    phase_events["event_timestamp_utc"] = pd.to_datetime(phase_events["event_timestamp_utc"], utc=True)
    distance_events["event_timestamp_utc"] = pd.to_datetime(distance_events["event_timestamp_utc"], utc=True)
    rows = []

    for _, phase in phase_events.iterrows():
        deltas = (distance_events["event_timestamp_utc"] - phase["event_timestamp_utc"]).abs()
        nearest_idx = deltas.idxmin()
        nearest = distance_events.loc[nearest_idx]
        days_apart = deltas.loc[nearest_idx] / pd.Timedelta(days=1)
        if days_apart <= threshold_days:
            rows.append(
                {
                    "event_timestamp_utc": phase["event_timestamp_utc"],
                    "event_date": phase["event_timestamp_utc"].date(),
                    "event_type": f"{phase['event_type']} near {nearest['event_type']}",
                    "phase_event_type": phase["event_type"],
                    "distance_event_type": nearest["event_type"],
                    "combo_partner_timestamp_utc": nearest["event_timestamp_utc"],
                    "days_apart": float(days_apart),
                    "moon_distance_km": nearest.get("moon_distance_km", np.nan),
                    "calculation_method": f"Phase-distance combo within {threshold_days} calendar days",
                }
            )

    combos = pd.DataFrame(rows)
    if combos.empty:
        return combos
    return map_events_to_trading_days(combos, trading_index, direction=mapping_direction)


def calculate_high_low_clusters(
    price_data: pd.DataFrame,
    mapped_events: pd.DataFrame,
    local_window: int = 20,
    proximity_window: int = 3,
    random_trials: int = 500,
    random_seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compare event proximity to local highs/lows against random sessions."""
    if price_data.empty or mapped_events.empty:
        return pd.DataFrame(), pd.DataFrame()

    prices = price_data[price_column(price_data)].dropna()
    sessions = pd.DatetimeIndex(prices.index).normalize()
    rolling_span = local_window * 2 + 1
    local_high = prices.eq(prices.rolling(rolling_span, center=True, min_periods=local_window + 1).max())
    local_low = prices.eq(prices.rolling(rolling_span, center=True, min_periods=local_window + 1).min())
    high_positions = np.flatnonzero(local_high.to_numpy())
    low_positions = np.flatnonzero(local_low.to_numpy())
    session_to_pos = {session: pos for pos, session in enumerate(sessions)}

    event_rows = []
    for _, event in mapped_events.dropna(subset=["mapped_market_date"]).iterrows():
        mapped_date = pd.to_datetime(event["mapped_market_date"]).normalize()
        if mapped_date not in session_to_pos:
            continue
        pos = session_to_pos[mapped_date]
        near_high = bool(np.any(np.abs(high_positions - pos) <= proximity_window))
        near_low = bool(np.any(np.abs(low_positions - pos) <= proximity_window))
        event_rows.append(
            {
                "event_type": event["event_type"],
                "event_timestamp_utc": event["event_timestamp_utc"],
                "mapped_market_date": mapped_date,
                "near_local_high": near_high,
                "near_local_low": near_low,
            }
        )

    event_list = pd.DataFrame(event_rows)
    if event_list.empty:
        return pd.DataFrame(), event_list

    rng = np.random.default_rng(random_seed)
    valid_positions = np.arange(len(sessions))
    baseline_rows = []
    for event_type, group in event_list.groupby("event_type", observed=True):
        n = len(group)
        random_high_rates = []
        random_low_rates = []
        for _ in range(random_trials):
            sampled = rng.choice(valid_positions, size=n, replace=n > len(valid_positions))
            random_high_rates.append(np.mean([np.any(np.abs(high_positions - pos) <= proximity_window) for pos in sampled]))
            random_low_rates.append(np.mean([np.any(np.abs(low_positions - pos) <= proximity_window) for pos in sampled]))

        baseline_rows.append(
            {
                "event_type": event_type,
                "total_events": n,
                "near_high_count": int(group["near_local_high"].sum()),
                "near_low_count": int(group["near_local_low"].sum()),
                "near_high_pct": group["near_local_high"].mean(),
                "near_low_pct": group["near_local_low"].mean(),
                "random_near_high_pct": float(np.mean(random_high_rates)),
                "random_near_low_pct": float(np.mean(random_low_rates)),
                "sample_quality": "ok" if n >= 10 else "insufficient sample size",
            }
        )

    summary = pd.DataFrame(baseline_rows)
    summary["difference_vs_random_high"] = summary["near_high_pct"] - summary["random_near_high_pct"]
    summary["difference_vs_random_low"] = summary["near_low_pct"] - summary["random_near_low_pct"]
    return summary.sort_values("event_type").reset_index(drop=True), event_list


def calculate_realized_volatility(price_data: pd.DataFrame, mapped_events: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Calculate annualized realized volatility before and after each mapped event."""
    if price_data.empty or mapped_events.empty:
        return pd.DataFrame(), pd.DataFrame()

    prices = price_data[price_column(price_data)].dropna().astype(float)
    log_returns = np.log(prices / prices.shift(1)).dropna()
    sessions = pd.DatetimeIndex(prices.index).normalize()
    session_to_pos = {session: pos for pos, session in enumerate(sessions)}
    rows = []

    for _, event in mapped_events.dropna(subset=["mapped_market_date"]).iterrows():
        mapped_date = pd.to_datetime(event["mapped_market_date"]).normalize()
        if mapped_date not in session_to_pos:
            continue
        pos = session_to_pos[mapped_date]
        for window in [5, 10]:
            before = log_returns.iloc[max(0, pos - window + 1) : pos + 1]
            after = log_returns.iloc[pos + 1 : pos + window + 1]
            if len(before) < max(2, window // 2) or len(after) < max(2, window // 2):
                continue
            before_vol = float(before.std(ddof=1) * np.sqrt(252))
            after_vol = float(after.std(ddof=1) * np.sqrt(252))
            rows.append(
                {
                    "event_type": event["event_type"],
                    "event_timestamp_utc": event["event_timestamp_utc"],
                    "mapped_market_date": mapped_date,
                    "window": window,
                    "volatility_before": before_vol,
                    "volatility_after": after_vol,
                    "volatility_change": after_vol - before_vol,
                }
            )

    event_level = pd.DataFrame(rows)
    if event_level.empty:
        return pd.DataFrame(), event_level

    summary = (
        event_level.groupby(["event_type", "window"], observed=True)
        .agg(
            average_volatility_before=("volatility_before", "mean"),
            average_volatility_after=("volatility_after", "mean"),
            average_volatility_change=("volatility_change", "mean"),
            median_volatility_change=("volatility_change", "median"),
            occurrences=("volatility_change", "count"),
        )
        .reset_index()
    )
    summary["sample_quality"] = np.where(summary["occurrences"] >= 10, "ok", "insufficient sample size")
    return summary.sort_values(["event_type", "window"]).reset_index(drop=True), event_level
