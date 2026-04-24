"""Six v1 Measure implementations. Each produces a SQL fragment and lists required columns."""

from sas.query.registry import register_measure


class CountOpen:
    name = "count_open"
    required_columns = ["state", "last_seen"]

    def build_select_sql(self, target_date: str) -> str:
        return (
            f"COUNT(*) FILTER (WHERE state = 'OPEN' AND CAST(last_seen AS DATE) <= '{target_date}')"
        )


class CountNew:
    name = "count_new"
    required_columns = ["state", "first_seen"]

    def build_select_sql(self, target_date: str) -> str:
        return (
            f"COUNT(*) FILTER (WHERE state = 'OPEN' AND CAST(first_seen AS DATE) = '{target_date}')"
        )


class CountFixed:
    name = "count_fixed"
    required_columns = ["state", "closed_at"]

    def build_select_sql(self, target_date: str) -> str:
        return (
            f"COUNT(*) FILTER (WHERE state = 'CLOSED' AND CAST(closed_at AS DATE) = '{target_date}')"
        )


class CountRegressed:
    name = "count_regressed"
    required_columns = ["reopened_at"]

    def build_select_sql(self, target_date: str) -> str:
        return (
            f"COUNT(*) FILTER (WHERE reopened_at IS NOT NULL AND CAST(reopened_at AS DATE) = '{target_date}')"
        )


class CountDistinctCve:
    name = "count_distinct_cve"
    required_columns = ["cve_id", "state"]

    def build_select_sql(self, target_date: str) -> str:
        return (
            f"COUNT(DISTINCT cve_id) FILTER (WHERE state = 'OPEN' AND CAST(last_seen AS DATE) <= '{target_date}')"
        )


class Mttr:
    name = "mttr"
    required_columns = ["days_open", "state"]

    def build_select_sql(self, target_date: str) -> str:
        return (
            f"AVG(days_open) FILTER (WHERE state = 'CLOSED' AND CAST(closed_at AS DATE) = '{target_date}')"
        )


# Registration — must run at import time
for _cls in [CountOpen, CountNew, CountFixed, CountRegressed, CountDistinctCve, Mttr]:
    register_measure(_cls.name, _cls)
