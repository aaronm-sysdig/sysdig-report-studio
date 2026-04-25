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


class CountOpenCritical:
    name = "count_open_critical"
    required_columns = ["count_open_critical"]

    def build_select_sql(self, target_date) -> str:
        return "count_open_critical"


class CountOpenHigh:
    name = "count_open_high"
    required_columns = ["count_open_high"]

    def build_select_sql(self, target_date) -> str:
        return "count_open_high"


class CountOpenMedium:
    name = "count_open_medium"
    required_columns = ["count_open_medium"]

    def build_select_sql(self, target_date) -> str:
        return "count_open_medium"


class CountOpenLow:
    name = "count_open_low"
    required_columns = ["count_open_low"]

    def build_select_sql(self, target_date) -> str:
        return "count_open_low"


class CountOpenNegligible:
    name = "count_open_negligible"
    required_columns = ["count_open_negligible"]

    def build_select_sql(self, target_date) -> str:
        return "count_open_negligible"


class CountFixedPatched:
    name = "count_fixed_patched"
    required_columns = ["count_fixed_patched"]

    def build_select_sql(self, target_date) -> str:
        return "count_fixed_patched"


class CountFixedRetired:
    name = "count_fixed_retired"
    required_columns = ["count_fixed_retired"]

    def build_select_sql(self, target_date) -> str:
        return "count_fixed_retired"


class CountFixedAccepted:
    name = "count_fixed_accepted"
    required_columns = ["count_fixed_accepted"]

    def build_select_sql(self, target_date) -> str:
        return "count_fixed_accepted"


class CountFixedOther:
    name = "count_fixed_other"
    required_columns = ["count_fixed_other"]

    def build_select_sql(self, target_date) -> str:
        return "count_fixed_other"


# Registration — must run at import time
for _cls in [
    CountOpen, CountNew, CountFixed, CountRegressed, CountDistinctCve, Mttr,
    CountOpenCritical, CountOpenHigh, CountOpenMedium, CountOpenLow,
    CountOpenNegligible,
    CountFixedPatched, CountFixedRetired, CountFixedAccepted, CountFixedOther,
]:
    register_measure(_cls.name, _cls)
