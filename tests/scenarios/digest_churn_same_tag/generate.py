"""Scenario: Same tag rebuilt daily -> 3 distinct digests, all with one CVE.

Expected: 3 image rows, 1 repository row, image_in_repository has 3 entries
all linking to the same repository/tag.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[3]))  # noqa: E402
from tests.scenarios._builder import ScenarioBuilder  # noqa: E402

HERE = Path(__file__).parent


def main():
    b = ScenarioBuilder()

    for day, (date, digest) in enumerate(
        [
            ("2026-05-01", "sha256:build1"),
            ("2026-05-02", "sha256:build2"),
            ("2026-05-03", "sha256:build3"),
        ],
        start=1,
    ):
        b.add_finding(
            vulnerability_name="CVE-2026-2001",
            image_id=digest,
            image_name="registry.example.com/myapp:latest",
            kubernetes_cluster_name="prod-cluster",
            kubernetes_namespace_name="default",
            kubernetes_workload_name="myapp",
        )
        b.write_csv(HERE / f"day{day}_{date}.csv")
        b.clear()


if __name__ == "__main__":
    main()
