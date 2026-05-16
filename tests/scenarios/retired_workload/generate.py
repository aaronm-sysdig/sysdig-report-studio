"""Scenario: Workload disappears on day 3; CVE closes after grace period.

Expected: CVE-2026-3001 OPEN days 1-2, STALE day 3-5, CLOSED/REMEDIED day 6
(grace period of 3 days expired).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[3]))  # noqa: E402
from tests.scenarios._builder import ScenarioBuilder  # noqa: E402

HERE = Path(__file__).parent

FILLER = dict(
    vulnerability_name="CVE-2026-3002",
    image_id="sha256:other1",
    image_name="registry.example.com/other-app:v1",
    kubernetes_cluster_name="prod-cluster",
    kubernetes_namespace_name="test-ns",
    kubernetes_workload_name="other-workload",
)


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

    # Day 3 — workload gone (STALE begins)
    b.add_finding(**FILLER)
    b.write_csv(HERE / "day3_2026-05-03.csv")
    b.clear()

    # Day 4-5 — still absent (still STALE, 1-2 days into grace)
    for day, date in enumerate(["2026-05-04", "2026-05-05"], start=4):
        b.add_finding(**FILLER)
        b.write_csv(HERE / f"day{day}_{date}.csv")
        b.clear()

    # Day 6 — still absent (grace period expired → CLOSED/REMEDIED)
    b.add_finding(**FILLER)
    b.write_csv(HERE / "day6_2026-05-06.csv")
    b.clear()


if __name__ == "__main__":
    main()
