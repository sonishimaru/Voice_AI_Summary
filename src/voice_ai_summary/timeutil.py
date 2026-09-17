"""Local-time helpers. All persisted timestamps are UTC ISO 8601 with a trailing `Z`;
this module is the only place that converts between that and a viewer's local timezone.
"""

from __future__ import annotations

import re
from datetime import UTC, datetime, timedelta
from zoneinfo import ZoneInfo

_HM_RE = re.compile(r"^(?P<h>\d{1,2}):(?P<m>\d{2})$")


def _parse_utc(iso_utc: str) -> datetime:
    """Parse a `YYYY-MM-DDTHH:MM:SSZ` string into an aware UTC datetime."""
    return datetime.strptime(iso_utc, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=UTC)


def _fmt_utc(dt: datetime) -> str:
    return dt.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def local_day_bounds(day: str, tz: str) -> tuple[str, str]:
    """UTC ISO `Z` strings for the [00:00, 24:00) window of local date `day` (`YYYY-MM-DD`)."""
    zone = ZoneInfo(tz)
    start_local = datetime.strptime(day, "%Y-%m-%d").replace(tzinfo=zone)
    end_local = start_local + timedelta(days=1)
    return _fmt_utc(start_local), _fmt_utc(end_local)


def to_local(iso_utc: str, tz: str) -> datetime:
    """Convert a stored UTC ISO timestamp to an aware datetime in `tz`."""
    return _parse_utc(iso_utc).astimezone(ZoneInfo(tz))


def fmt_hm(iso_utc: str, tz: str) -> str:
    """Render a stored UTC ISO timestamp as local `HH:MM`."""
    return to_local(iso_utc, tz).strftime("%H:%M")


def today_local(tz: str) -> str:
    """Today's date (`YYYY-MM-DD`) in `tz`."""
    return datetime.now(ZoneInfo(tz)).strftime("%Y-%m-%d")


def local_time_to_utc(day: str, hm: str, tz: str) -> str:
    """Convert `HH:MM` on local date `day` (`YYYY-MM-DD`) in `tz` to a UTC ISO `Z` string.

    `24:00` is accepted and means midnight at the *start* of the next local day (the
    usual convention for writing a half-open range's exclusive end as, e.g., `23:45` to
    `24:00`). Raises `ValueError` on a malformed `day` or `hm`, or an hour/minute out of
    the ordinary `00:00`-`23:59` range (`24:00` itself excepted).
    """
    match = _HM_RE.match(hm)
    if not match:
        raise ValueError(f"not a HH:MM time: {hm!r}")
    hour, minute = int(match["h"]), int(match["m"])
    if minute > 59:
        raise ValueError(f"not a HH:MM time: {hm!r}")

    zone = ZoneInfo(tz)
    if hour == 24 and minute == 0:
        start_local = datetime.strptime(day, "%Y-%m-%d").replace(tzinfo=zone) + timedelta(days=1)
    elif 0 <= hour <= 23:
        start_local = datetime.strptime(day, "%Y-%m-%d").replace(
            tzinfo=zone, hour=hour, minute=minute
        )
    else:
        raise ValueError(f"not a HH:MM time: {hm!r}")
    return _fmt_utc(start_local)
