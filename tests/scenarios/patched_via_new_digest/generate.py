"""Scenario: CVE fixed by upgrading to a new image digest (v1->v2 drops CVE-2026-1001).

Expected: CVE-2026-1001 enters STALE on day 3 (v1 gone), closes after grace period
with reason_code=REMEDIED. CVE-2026-1002 stays OPEN on v2.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[3]))  # noqa: E402
from tests.scenarios._builder import ScenarioBuilder  # noqa: E402

HERE = Path(__file__).parent


def main():
    b = ScenarioBuilder()

    # Day 1 — v1 image has both CVEs
    for cve in ("CVE-2026-1001", "CVE-2026-1002"):
        b.add_finding(
            vulnerability_name=cve,
            image_id="sha256:v1-digest",
            image_name="registry.example.com/myapp:v1.0",
            kubernetes_cluster_name="prod-cluster",
            kubernetes_namespace_name="default",
            kubernetes_workload_name="myapp",
        )
    b.write_csv(HERE / "day1_2026-05-01.csv")
    b.clear()

    # Day 2 — same image, same CVEs (re-seen)
    for cve in ("CVE-2026-1001", "CVE-2026-1002"):
        b.add_finding(
            vulnerability_name=cve,
            image_id="sha256:v1-digest",
            image_name="registry.example.com/myapp:v1.0",
            kubernetes_cluster_name="prod-cluster",
            kubernetes_namespace_name="default",
            kubernetes_workload_name="myapp",
        )
    b.write_csv(HERE / "day2_2026-05-02.csv")
    b.clear()

    # Day 3 — v2 image only has CVE-2026-1002; v1 is gone (STALE begins)
    b.add_finding(
        vulnerability_name="CVE-2026-1002",
        image_id="sha256:v2-digest",
        image_name="registry.example.com/myapp:v2.0",
        kubernetes_cluster_name="prod-cluster",
        kubernetes_namespace_name="default",
        kubernetes_workload_name="myapp",
    )
    b.write_csv(HERE / "day3_2026-05-03.csv")
    b.clear()

    # Day 4-5 — v2 still present (STALE continues, 1-2 days into grace)
    for day, date in enumerate(["2026-05-04", "2026-05-05"], start=4):
        b.add_finding(
            vulnerability_name="CVE-2026-1002",
            image_id="sha256:v2-digest",
            image_name="registry.example.com/myapp:v2.0",
            kubernetes_cluster_name="prod-cluster",
            kubernetes_namespace_name="default",
            kubernetes_workload_name="myapp",
        )
        b.write_csv(HERE / f"day{day}_{date}.csv")
        b.clear()

    # Day 6 — grace period expired, CVE-2026-1001 closes
    b.add_finding(
        vulnerability_name="CVE-2026-1002",
        image_id="sha256:v2-digest",
        image_name="registry.example.com/myapp:v2.0",
        kubernetes_cluster_name="prod-cluster",
        kubernetes_namespace_name="default",
        kubernetes_workload_name="myapp",
    )
    b.write_csv(HERE / "day6_2026-05-06.csv")
    b.clear()


if __name__ == "__main__":
    main()
