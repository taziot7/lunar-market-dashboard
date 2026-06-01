from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import find_peaks
from skyfield import almanac
from skyfield.api import Loader


DATA_DIR = Path(__file__).resolve().parents[1] / "data"
SKYFIELD_DIR = DATA_DIR / "skyfield"
PHASE_NAMES = {
    0: "New Moon",
    1: "First Quarter",
    2: "Full Moon",
    3: "Last Quarter",
}


def _skyfield_loader() -> Loader:
    SKYFIELD_DIR.mkdir(parents=True, exist_ok=True)
    return Loader(str(SKYFIELD_DIR))


def _load_ephemeris():
    load = _skyfield_loader()
    return load, load("de421.bsp")


def calculate_lunar_phases(start_date: date, end_date: date) -> pd.DataFrame:
    """Calculate new and full moons with Skyfield's phase search."""
    load, eph = _load_ephemeris()
    ts = load.timescale()
    start = pd.to_datetime(start_date).date() - timedelta(days=2)
    end = pd.to_datetime(end_date).date() + timedelta(days=2)

    t0 = ts.utc(start.year, start.month, start.day)
    t1 = ts.utc(end.year, end.month, end.day)
    times, phases = almanac.find_discrete(t0, t1, almanac.moon_phases(eph))

    rows = []
    for time, phase_code in zip(times, phases):
        event_type = PHASE_NAMES[int(phase_code)]
        if event_type not in {"New Moon", "Full Moon"}:
            continue
        event_ts = pd.Timestamp(time.utc_datetime()).tz_convert("UTC")
        if pd.to_datetime(start_date).date() <= event_ts.date() <= pd.to_datetime(end_date).date():
            rows.append(
                {
                    "event_timestamp_utc": event_ts,
                    "event_date": event_ts.date(),
                    "event_type": event_type,
                    "moon_distance_km": np.nan,
                    "calculation_method": "Skyfield moon phase",
                }
            )

    return pd.DataFrame(rows)


def calculate_lunar_distance_events(start_date: date, end_date: date, sample_hours: int = 6) -> pd.DataFrame:
    """Approximate apogee/perigee by detecting local extrema in sampled Moon distance.

    Skyfield provides accurate Earth-Moon distances for each sample. The exact
    apogee/perigee instant is approximated by the nearest sampled local maximum
    or minimum, which is sufficient for daily market-session research.
    """
    load, eph = _load_ephemeris()
    ts = load.timescale()

    start = pd.to_datetime(start_date).normalize() - pd.Timedelta(days=30)
    end = pd.to_datetime(end_date).normalize() + pd.Timedelta(days=30)
    sample_index = pd.date_range(start=start, end=end, freq=f"{sample_hours}h", tz="UTC")
    if sample_index.empty:
        return pd.DataFrame()

    times = ts.utc(
        sample_index.year.to_numpy(),
        sample_index.month.to_numpy(),
        sample_index.day.to_numpy(),
        sample_index.hour.to_numpy(),
        sample_index.minute.to_numpy(),
    )
    earth = eph["earth"]
    moon = eph["moon"]
    distances = earth.at(times).observe(moon).distance().km

    min_same_type_spacing = max(1, int((20 * 24) / sample_hours))
    apogee_idx, _ = find_peaks(distances, distance=min_same_type_spacing, prominence=8_000)
    perigee_idx, _ = find_peaks(-distances, distance=min_same_type_spacing, prominence=8_000)

    rows = []
    range_start = pd.to_datetime(start_date).date()
    range_end = pd.to_datetime(end_date).date()
    for event_type, indexes in [("Apogee", apogee_idx), ("Perigee", perigee_idx)]:
        for idx in indexes:
            event_ts = pd.Timestamp(sample_index[idx]).tz_convert("UTC")
            if range_start <= event_ts.date() <= range_end:
                rows.append(
                    {
                        "event_timestamp_utc": event_ts,
                        "event_date": event_ts.date(),
                        "event_type": event_type,
                        "moon_distance_km": float(distances[idx]),
                        "calculation_method": f"Skyfield distance sampled every {sample_hours}h",
                    }
                )

    return pd.DataFrame(rows)


def calculate_lunar_events(start_date: date, end_date: date) -> pd.DataFrame:
    phase_events = calculate_lunar_phases(start_date, end_date)
    distance_events = calculate_lunar_distance_events(start_date, end_date)
    events = pd.concat([phase_events, distance_events], ignore_index=True)
    if events.empty:
        return events
    events["event_timestamp_utc"] = pd.to_datetime(events["event_timestamp_utc"], utc=True)
    events["event_date"] = events["event_timestamp_utc"].dt.date
    return events.sort_values(["event_timestamp_utc", "event_type"]).reset_index(drop=True)
