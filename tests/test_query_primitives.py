from datetime import date
from sas.query.primitives import (
    Query, QueryResult, Series, TimeWindow, Filter, Ordering
)


def test_timewindow_last_n():
    tw = TimeWindow(mode="last_n_snapshots", n=30, granularity="day")
    assert tw.n == 30
    assert tw.start is None


def test_timewindow_date_range():
    tw = TimeWindow(mode="date_range", start=date(2026, 1, 1), end=date(2026, 4, 1), granularity="week")
    assert tw.start < tw.end


def test_filter_equality():
    f = Filter(field="severity", operator="eq", value="Critical")
    assert f.field == "severity"
    assert f.value == "Critical"


def test_query_minimal():
    tw = TimeWindow(mode="last_n_snapshots", n=7, granularity="day")
    q = Query(lens="Image", traversal=[], time=tw, measure="count_open", filters=[])
    assert q.lens == "Image"
    assert q.limit is None


def test_query_result_empty():
    qr = QueryResult(series=[], dimensions={}, snapshot_range=(date(2026, 1, 1), date(2026, 4, 1)), missing_days=[], exec_time_ms=0)
    assert qr.series == []


def test_series_shape():
    s = Series(key={"severity": "Critical"}, x=[date(2026, 4, 1)], y=[42])
    assert len(s.x) == len(s.y)


def test_ordering():
    o = Ordering(field="date", direction="asc")
    assert o.direction in ("asc", "desc")
