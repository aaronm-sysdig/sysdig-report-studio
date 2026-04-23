"""Scenario: Image name contains a registry port (host:port/path:tag).

Expected: _split_image_name handles the colon-in-host correctly; image is
queryable and image_in_repository has a row. May expose parsing edge cases.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[3]))  # noqa: E402
from tests.scenarios._builder import ScenarioBuilder  # noqa: E402

HERE = Path(__file__).parent


def main():
    b = ScenarioBuilder()

    common = dict(
        vulnerability_name="CVE-2026-7001",
        image_id="sha256:ecr1",
        image_name="registry.internal:5000/team/app:v1",
        kubernetes_cluster_name="prod-cluster",
        kubernetes_namespace_name="team-ns",
        kubernetes_workload_name="team-app",
    )

    b.add_finding(**common)
    b.write_csv(HERE / "day1_2026-05-01.csv")
    b.clear()

    b.add_finding(**common)
    b.write_csv(HERE / "day2_2026-05-02.csv")
    b.clear()


if __name__ == "__main__":
    main()
