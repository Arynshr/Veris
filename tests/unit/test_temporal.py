from datetime import date

import pytest

from veris.analysis.temporal import resolve_dates, text_dates, url_date, window_start
from veris.core.models import DateSource

AS_OF = date(2026, 1, 1)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("On March 3, 2021 he was charged.", {date(2021, 3, 3)}),
        ("On 5 May 2022 the case closed.", {date(2022, 5, 5)}),
        ("Filed 2020-01-02 in court.", {date(2020, 1, 2)}),
        ("Sept. 9th, 2019 ruling", {date(2019, 9, 9)}),
        ("Convicted in 2021.", {date(2021, 1, 1)}),
        ("Revenue grew 1999 percent", set()),
        ("February 30, 2021 is not real", set()),
        ("no dates here", set()),
        ("", set()),
    ],
)
def test_text_dates(text, expected):
    assert set(text_dates(text)) == expected


@pytest.mark.parametrize(
    ("url", "expected"),
    [
        ("https://x.com/news/2023/05/12/story", date(2023, 5, 12)),
        ("https://x.com/2023/05/story", date(2023, 5, 1)),
        ("https://x.com/news/story-1234", None),
        ("https://x.com/2023/13/12/story", None),
    ],
)
def test_url_date(url, expected):
    assert url_date(url) == expected


def test_source_precedence_metadata_then_text_then_url():
    url = "https://x.com/2020/01/02/a"
    assert (
        resolve_dates("Convicted in 2021.", url, date(2024, 2, 2), AS_OF).date_source
        == DateSource.METADATA
    )
    text = resolve_dates("Convicted in 2021.", url, None, AS_OF)
    assert text.date_source == DateSource.TEXT and text.event_date == date(2021, 1, 1)
    via_url = resolve_dates("nothing here", url, None, AS_OF)
    assert via_url.date_source == DateSource.URL and via_url.published_date == date(
        2020, 1, 2
    )
    assert (
        resolve_dates("nothing here", "https://x.com/a", None, AS_OF).date_source
        == DateSource.NONE
    )


def test_latest_text_date_wins_and_future_dates_are_ignored():
    info = resolve_dates(
        "Charged on 2019-03-01, convicted on 2021-06-10, appeal due 2030-01-01.",
        "https://x.com/a",
        None,
        AS_OF,
    )
    assert info.event_date == date(2021, 6, 10)


@pytest.mark.parametrize("years", [1, 2, 5])
def test_window_start_scales_with_lookback(years):
    start = window_start(AS_OF, years)
    assert abs((AS_OF - start).days - round(365.25 * years)) <= 1
    assert start < AS_OF
