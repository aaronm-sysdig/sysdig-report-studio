"""Scenario: Workload disappears on day 3; CVE closes with reason_code=RETIRED.

Expected: CVE-2026-3001 is OPEN days 1-2, then closes on day 3 because the
image is no longer running anywhere.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[3]))  # noqa: E402
from tests.scenarios._builder import ScenarioBuilder  # noqa: E402

HERE = Path(__file__).parent


def main():
    b = ScenarioBuilder()

    for day, date in enumerate(["2026-05-01", "2026-05-02"], start=1):
        b.add_finding(
            vulnerability_name="CVE-2026-3001",
            image_id="sha256:orphan1",
            image_name="registry.example.com/orphan-app:v1",
            kubernetes_cluster_name="prod-cluster",
            kubernetes_namespace_name="test-ns",
            kubernetes_workload_name="orphan-workload",
        )
        b.write_csv(HERE / f"day{day}_{date}.csv")
        b.clear()

    # Day 3 — workload gone; different finding keeps CSV non-empty
    b.add_finding(
        vulnerability_name="CVE-2026-3002",
        image_id="sha256:other1",
        image_name="registry.example.com/other-app:v1",
        kubernetes_cluster_name="prod-cluster",
        kubernetes_namespace_name="test-ns",
        kubernetes_workload_name="other-workload",
    )
    b.write_csv(HERE / "day3_2026-05-03.csv")
    b.clear()


if __name__ == "__main__":
    main()
