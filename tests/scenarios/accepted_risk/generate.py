"""Scenario: risk_accepted flips false->true on day 2 (re-seen, flag updated).

Expected: after day 2 ingest, finding_state.risk_accepted is true for
CVE-2026-4001. Finding is NOT closed - it is re-seen with the flag changed.
If it then disappears on day 3, reason_code=ACCEPTED may not fire because
there is no today's row to compare against; code will likely fall back to
UNKNOWN. This scenario tests that known limitation.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[3]))  # noqa: E402
from tests.scenarios._builder import ScenarioBuilder  # noqa: E402

HERE = Path(__file__).parent


def main():
    b = ScenarioBuilder()

    # Day 1 — risk not yet accepted
    b.add_finding(
        vulnerability_name="CVE-2026-4001",
        image_id="sha256:accepted1",
        image_name="registry.example.com/webapp:v1",
        kubernetes_cluster_name="prod-cluster",
        kubernetes_namespace_name="app-ns",
        kubernetes_workload_name="accepted-workload",
        risk_accepted="false",
    )
    b.write_csv(HERE / "day1_2026-05-01.csv")
    b.clear()

    # Day 2 — same natural key, risk_accepted flipped to true
    b.add_finding(
        vulnerability_name="CVE-2026-4001",
        image_id="sha256:accepted1",
        image_name="registry.example.com/webapp:v1",
        kubernetes_cluster_name="prod-cluster",
        kubernetes_namespace_name="app-ns",
        kubernetes_workload_name="accepted-workload",
        risk_accepted="true",
    )
    b.write_csv(HERE / "day2_2026-05-02.csv")
    b.clear()

    # Day 3 — finding disappears; reason_code will likely be UNKNOWN
    b.add_finding(
        vulnerability_name="CVE-2026-4002",
        image_id="sha256:other-app1",
        image_name="registry.example.com/other-app:v1",
        kubernetes_cluster_name="prod-cluster",
        kubernetes_namespace_name="app-ns",
        kubernetes_workload_name="other-workload",
    )
    b.write_csv(HERE / "day3_2026-05-03.csv")
    b.clear()


if __name__ == "__main__":
    main()
