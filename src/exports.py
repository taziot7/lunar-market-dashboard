from __future__ import annotations

import pandas as pd


def dataframe_to_csv_bytes(frame: pd.DataFrame) -> bytes:
    if frame is None or frame.empty:
        return b""
    return frame.to_csv(index=False).encode("utf-8")


def tradingview_date_list(events: pd.DataFrame) -> pd.DataFrame:
    if events.empty:
        return pd.DataFrame(columns=["event_type", "event_date", "mapped_market_date", "timestamp_utc"])
    out = events.copy()
    out["timestamp_utc"] = pd.to_datetime(out["event_timestamp_utc"], utc=True).dt.strftime("%Y-%m-%d %H:%M:%S UTC")
    out["event_date"] = pd.to_datetime(out["event_timestamp_utc"], utc=True).dt.strftime("%Y-%m-%d")
    if "mapped_market_date" in out.columns:
        out["mapped_market_date"] = pd.to_datetime(out["mapped_market_date"]).dt.strftime("%Y-%m-%d")
    else:
        out["mapped_market_date"] = ""
    return out[["event_type", "event_date", "mapped_market_date", "timestamp_utc"]]


def generate_pine_script(events: pd.DataFrame, script_name: str = "Lunar Event Dates") -> str:
    if events.empty:
        return "// No lunar events available for the current filters."

    safe_name = script_name.replace('"', "")
    lines = [
        "//@version=5",
        f'indicator("{safe_name}", overlay=true, max_lines_count=500, max_labels_count=500)',
        "",
        "var eventTimes = array.new_int()",
        "var eventLabels = array.new_string()",
        "",
        "if barstate.isfirst",
    ]

    for _, row in events.sort_values("event_timestamp_utc").iterrows():
        ts = pd.to_datetime(row["event_timestamp_utc"], utc=True)
        label = str(row["event_type"]).replace('"', "")
        lines.append(
            f'    array.push(eventTimes, timestamp("UTC", {ts.year}, {ts.month}, {ts.day}, {ts.hour}, {ts.minute}))'
        )
        lines.append(f'    array.push(eventLabels, "{label}")')

    lines.extend(
        [
            "",
            "for i = 0 to array.size(eventTimes) - 1",
            "    eventTime = array.get(eventTimes, i)",
            "    if time >= eventTime and time[1] < eventTime",
            "        label.new(bar_index, high, array.get(eventLabels, i), style=label.style_label_down, textcolor=color.white, color=color.new(color.gray, 20))",
        ]
    )
    return "\n".join(lines)
