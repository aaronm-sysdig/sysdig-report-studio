import pytest
from sas.query.registry import LENSES, MEASURES, EDGES
from sas.query.measures import (
    CountOpen, CountNew, CountFixed, CountRegressed, CountDistinctCve, Mttr
)


def test_all_v1_lenses_registered():
    expected = {"Image", "CVE", "Workload", "Cluster", "Namespace", "Package", "Repository", "Team", "Owner"}
    assert expected == set(LENSES.keys())


def test_all_v1_measures_registered():
    expected = {
        "count_open", "count_new", "count_fixed", "count_regressed", "count_distinct_cve", "mttr",
        "count_open_critical", "count_open_high", "count_open_medium", "count_open_low",
        # Phase 2.3 — negligible tier + reason-code decomposition
        "count_open_negligible",
        "count_fixed_patched", "count_fixed_retired", "count_fixed_accepted", "count_fixed_other",
    }
    assert expected == set(MEASURES.keys())


def test_count_open_sql():
    m = CountOpen()
    sql = m.build_select_sql(target_date="2026-04-01")
    assert "state" in sql and "OPEN" in sql


def test_count_new_sql():
    m = CountNew()
    sql = m.build_select_sql(target_date="2026-04-01")
    assert "first_seen" in sql


def test_count_fixed_sql():
    m = CountFixed()
    sql = m.build_select_sql(target_date="2026-04-01")
    assert "closed_at" in sql


def test_count_regressed_sql():
    m = CountRegressed()
    sql = m.build_select_sql(target_date="2026-04-01")
    assert "reopened_at" in sql


def test_count_distinct_cve_sql():
    m = CountDistinctCve()
    sql = m.build_select_sql(target_date="2026-04-01")
    assert "DISTINCT" in sql and "cve_id" in sql


def test_mttr_sql():
    m = Mttr()
    sql = m.build_select_sql(target_date="2026-04-01")
    assert "AVG" in sql and "days_open" in sql


def test_all_measures_have_required_columns():
    for name, cls in [
        ("count_open", CountOpen), ("count_new", CountNew), ("count_fixed", CountFixed),
        ("count_regressed", CountRegressed), ("count_distinct_cve", CountDistinctCve), ("mttr", Mttr),
    ]:
        m = cls()
        assert isinstance(m.required_columns, list)
        assert len(m.required_columns) > 0


def test_count_open_critical_measure_registered():
    from sas.query.registry import MEASURES
    assert "count_open_critical" in MEASURES


def test_count_open_high_measure_registered():
    from sas.query.registry import MEASURES
    assert "count_open_high" in MEASURES


def test_count_open_medium_measure_registered():
    from sas.query.registry import MEASURES
    assert "count_open_medium" in MEASURES


def test_count_open_low_measure_registered():
    from sas.query.registry import MEASURES
    assert "count_open_low" in MEASURES


def test_count_open_critical_builds_correct_sql():
    from sas.query.registry import MEASURES
    from datetime import date
    m = MEASURES["count_open_critical"]()
    sql = m.build_select_sql(date(2026, 4, 23))
    assert "count_open_critical" in sql.lower()


def test_count_open_negligible_measure_registered():
    from sas.query.registry import MEASURES
    assert "count_open_negligible" in MEASURES


def test_count_fixed_patched_measure_registered():
    from sas.query.registry import MEASURES
    assert "count_fixed_patched" in MEASURES


def test_count_fixed_retired_measure_registered():
    from sas.query.registry import MEASURES
    assert "count_fixed_retired" in MEASURES


def test_count_fixed_accepted_measure_registered():
    from sas.query.registry import MEASURES
    assert "count_fixed_accepted" in MEASURES


def test_count_fixed_other_measure_registered():
    from sas.query.registry import MEASURES
    assert "count_fixed_other" in MEASURES
