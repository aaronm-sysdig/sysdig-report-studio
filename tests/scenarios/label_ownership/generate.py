"""Scenario: Namespace labels drive team ownership resolution.

Expected: with LabelStrategy("team") first in the resolver chain,
workload_owned_by.team_id="team-alpha" and
resolved_by_strategy="label:team".
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[3]))  # noqa: E402
from tests.scenarios._builder import ScenarioBuilder  # noqa: E402

HERE = Path(__file__).parent

_LABELS = (
    '{"kubernetes.namespace.label.team":"team-alpha",'
    '"kubernetes.namespace.label.owner":"alice"}'
)


def main():
    b = ScenarioBuilder()

    common = dict(
        vulnerability_name="CVE-2026-8001",
        image_id="sha256:labeled1",
        image_name="registry.example.com/team-alpha-app:v1",
        kubernetes_cluster_name="prod-cluster",
        kubernetes_namespace_name="team-alpha-ns",
        kubernetes_workload_name="labeled-app",
        namespace_labels=_LABELS,
    )

    b.add_finding(**common)
    b.write_csv(HERE / "day1_2026-05-01.csv")
    b.clear()

    b.add_finding(**common)
    b.write_csv(HERE / "day2_2026-05-02.csv")
    b.clear()


if __name__ == "__main__":
    main()
