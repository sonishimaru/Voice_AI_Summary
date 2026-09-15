"""Local-time helpers. All persisted timestamps are UTC ISO 8601 with a trailing `Z`;
this module is the only place that converts between that and a viewer's local timezone.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from zoneinfo import ZoneInfo


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
