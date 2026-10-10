"""Date extraction from passage text / URL path and lookback-window helpers."""

import re
from datetime import date, timedelta

from pydantic import BaseModel

from veris.core.models import DateSource

_MONTHS = {
    m: i
    for i, m in enumerate(
        [
            "jan",
            "feb",
            "mar",
            "apr",
            "may",
            "jun",
            "jul",
            "aug",
            "sep",
            "oct",
            "nov",
            "dec",
        ],
        start=1,
    )
}
_MON = r"(Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:t(?:ember)?)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)"
_YEAR = r"((?:19|20)\d{2})"
_ISO_RE = re.compile(rf"\b{_YEAR}-(0[1-9]|1[0-2])-(0[1-9]|[12]\d|3[01])\b")
_MDY_RE = re.compile(
    rf"\b{_MON}\.?\s+(?:(\d{{1,2}})(?:st|nd|rd|th)?,?\s+)?{_YEAR}\b", re.IGNORECASE
)
_DMY_RE = re.compile(
    rf"\b(\d{{1,2}})(?:st|nd|rd|th)?\s+{_MON}\.?,?\s+{_YEAR}\b", re.IGNORECASE
)
_YEAR_ONLY_RE = re.compile(rf"\b(?:in|during|since)\s+{_YEAR}\b", re.IGNORECASE)
_URL_RE = re.compile(
    r"/((?:19|20)\d{2})/(0[1-9]|1[0-2])(?:/(0[1-9]|[12]\d|3[01]))?(?=[/\-_.]|$)"
)


class DateInfo(BaseModel):
    event_date: date | None = None
    published_date: date | None = None
    date_source: DateSource = DateSource.NONE


def _safe(year: int, month: int, day: int = 1) -> date | None:
    try:
        return date(year, month, day)
    except ValueError:
        return None


def text_dates(text: str) -> list[date]:
    found: list[date | None] = []
    found += [_safe(int(y), int(m), int(d)) for y, m, d in _ISO_RE.findall(text)]
    found += [
        _safe(int(y), _MONTHS[mon[:3].lower()], int(d))
        for d, mon, y in _DMY_RE.findall(text)
    ]
    remainder = _DMY_RE.sub(" ", text)  # avoid re-reading "5 May 2022" as "May 2022"
    found += [
        _safe(int(y), _MONTHS[mon[:3].lower()], int(d or 1))
        for mon, d, y in _MDY_RE.findall(remainder)
    ]
    found += [_safe(int(y), 1, 1) for y in _YEAR_ONLY_RE.findall(text)]
    return [d for d in found if d]


def url_date(url: str) -> date | None:
    m = _URL_RE.search(url)
    return _safe(int(m[1]), int(m[2]), int(m[3] or 1)) if m else None


def resolve_dates(
    passage: str, url: str, published_at: date | None, as_of: date
) -> DateInfo:
    """metadata > passage text > URL path; text/URL are used only when published_at is null."""
    if published_at:
        return DateInfo(
            event_date=published_at,
            published_date=published_at,
            date_source=DateSource.METADATA,
        )
    if dates := [d for d in text_dates(passage) if d <= as_of]:
        return DateInfo(event_date=max(dates), date_source=DateSource.TEXT)
    if d := url_date(url):
        return DateInfo(event_date=d, published_date=d, date_source=DateSource.URL)
    return DateInfo()


def window_start(as_of: date, lookback_years: int) -> date:
    return as_of - timedelta(days=round(365.25 * lookback_years))
