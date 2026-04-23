"""Scenario: ~200 rows across 20 images x 10 CVEs, 3 namespaces, 2 clusters.

Expected: bulk ingest completes in under 10 seconds (soft smoke test).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[3]))  # noqa: E402
from tests.scenarios._builder import ScenarioBuilder  # noqa: E402

HERE = Path(__file__).parent

_CLUSTERS = ["prod-cluster", "staging-cluster"]
_NAMESPACES = ["default", "backend-ns", "frontend-ns"]


def main():
    b = ScenarioBuilder()

    for img_idx in range(20):
        cluster = _CLUSTERS[img_idx % len(_CLUSTERS)]
        ns = _NAMESPACES[img_idx % len(_NAMESPACES)]
        digest = f"sha256:bulk-image-{img_idx:03d}"
        image_name = f"registry.example.com/bulk-app-{img_idx:02d}:v1"
        workload = f"bulk-workload-{img_idx:02d}"

        for cve_idx in range(10):
            b.add_finding(
                vulnerability_name=f"CVE-2026-9{img_idx:02d}{cve_idx}",
                image_id=digest,
                image_name=image_name,
                kubernetes_cluster_name=cluster,
                kubernetes_namespace_name=ns,
                kubernetes_workload_name=workload,
                package_name=f"libpkg-{cve_idx}",
                package_version=f"1.{cve_idx}.0",
            )

    b.write_csv(HERE / "day1_2026-05-01.csv")
    b.clear()


if __name__ == "__main__":
    main()
