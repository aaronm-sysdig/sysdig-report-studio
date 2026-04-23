"""Scenario: Same image spreads from prod-cluster to staging-cluster on day 2.

Expected: workload_runs_image_daily has entries for both clusters on day 2;
daily_metrics_by_cluster_severity has rows for both clusters.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[3]))  # noqa: E402
from tests.scenarios._builder import ScenarioBuilder  # noqa: E402

HERE = Path(__file__).parent


def main():
    b = ScenarioBuilder()

    # Day 1 — only prod-cluster
    b.add_finding(
        vulnerability_name="CVE-2026-6001",
        image_id="sha256:shared1",
        image_name="registry.example.com/shared-app:v1",
        kubernetes_cluster_name="prod-cluster",
        kubernetes_namespace_name="prod-ns",
        kubernetes_workload_name="prod-app",
    )
    b.write_csv(HERE / "day1_2026-05-01.csv")
    b.clear()

    # Day 2 — prod-cluster AND staging-cluster
    for cluster, ns, workload in [
        ("prod-cluster", "prod-ns", "prod-app"),
        ("staging-cluster", "staging-ns", "staging-app"),
    ]:
        b.add_finding(
            vulnerability_name="CVE-2026-6001",
            image_id="sha256:shared1",
            image_name="registry.example.com/shared-app:v1",
            kubernetes_cluster_name=cluster,
            kubernetes_namespace_name=ns,
            kubernetes_workload_name=workload,
        )
    b.write_csv(HERE / "day2_2026-05-02.csv")
    b.clear()


if __name__ == "__main__":
    main()
