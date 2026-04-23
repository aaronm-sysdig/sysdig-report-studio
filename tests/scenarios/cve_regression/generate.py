"""Scenario: CVE disappears then reappears - tests regression/reopen tracking.

Expected: CVE-2026-5001 OPEN day 1, CLOSED day 2, REOPENED day 3
(reopen_count=1, is_regression=true), re-seen day 4. Final state has a
closed row and an open row with reopen_count=1.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[3]))  # noqa: E402
from tests.scenarios._builder import ScenarioBuilder  # noqa: E402

HERE = Path(__file__).parent


def main():
    b = ScenarioBuilder()

    common = dict(
        vulnerability_name="CVE-2026-5001",
        image_id="sha256:regress1",
        image_name="registry.example.com/flaky-app:v1",
        kubernetes_cluster_name="prod-cluster",
        kubernetes_namespace_name="test-ns",
        kubernetes_workload_name="flaky-app",
    )

    # Day 1 — CVE present (OPEN)
    b.add_finding(**common)
    b.write_csv(HERE / "day1_2026-05-01.csv")
    b.clear()

    # Day 2 — CVE absent; filler row keeps CSV non-empty (CVE closes)
    b.add_finding(
        vulnerability_name="CVE-2026-5099",
        image_id="sha256:stable1",
        image_name="registry.example.com/stable-app:v1",
        kubernetes_cluster_name="prod-cluster",
        kubernetes_namespace_name="test-ns",
        kubernetes_workload_name="stable-app",
    )
    b.write_csv(HERE / "day2_2026-05-02.csv")
    b.clear()

    # Day 3 — CVE reappears (REOPENED)
    b.add_finding(**common)
    b.write_csv(HERE / "day3_2026-05-03.csv")
    b.clear()

    # Day 4 — CVE still present (re-seen)
    b.add_finding(**common)
    b.write_csv(HERE / "day4_2026-05-04.csv")
    b.clear()


if __name__ == "__main__":
    main()
