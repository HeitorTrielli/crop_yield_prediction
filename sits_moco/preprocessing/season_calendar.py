"""Season calendar helpers for Paraná soy daily .npy DOY channels.

Day 1 of a season folder ``YYYY-YYYY`` is 1 October of the planting year.
"""

from __future__ import annotations

import re
from datetime import date

# Default used by the legacy monthly tiled pipeline only.
SEASON_START_DATE = date(2022, 10, 2)


def season_start_from_year_range(year_range: str) -> date:
    """
    From folder label YYYY-YYYY (e.g. 2020-2021), return Oct 1 of the planting year.
    DOY channel uses (date - that day).days + 1 for daily .npy preprocessing.
    """
    s = year_range.strip()
    m = re.match(r"^(\d{4})-(\d{4})$", s)
    if not m:
        raise ValueError(f"Expected year-range like 2020-2021, got {year_range!r}")
    y1 = int(m.group(1))
    return date(y1, 10, 1)


def date_to_season_doy(d: date, season_start: date | None = None) -> int:
    """Day index in season: season_start -> 1, next calendar day -> 2."""
    start = season_start if season_start is not None else SEASON_START_DATE
    return (d - start).days + 1


def month_year_to_season_doy(season_year: int, month: int) -> int:
    """Map (season_year, month) to season DOY. Oct–Dec use season_year, Jan–Mar use season_year+1."""
    calendar_year = season_year if month >= 10 else season_year + 1
    d = date(calendar_year, month, 1)
    if d < SEASON_START_DATE:
        d = SEASON_START_DATE
    return date_to_season_doy(d)
