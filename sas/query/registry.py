"""Lens, Measure, and Edge registries. Dicts, not enums — additive by design."""

# Keys are the canonical string names used in Query.lens, Query.measure, Query.traversal.

LENSES: dict[str, dict] = {
    "Image":      {"primary_table": "image",      "pk": "image_id"},
    "CVE":        {"primary_table": "cve",         "pk": "cve_id"},
    "Workload":   {"primary_table": "workload",    "pk": ("cluster_name", "namespace_name", "workload_type", "workload_name")},
    "Cluster":    {"primary_table": "cluster",     "pk": "cluster_name"},
    "Namespace":  {"primary_table": "namespace",   "pk": ("cluster_name", "namespace_name")},
    "Package":    {"primary_table": "package",     "pk": ("package_name", "package_type")},
    "Repository": {"primary_table": "repository",  "pk": "repository"},
    "Owner":      {"primary_table": "owner",       "pk": "owner_id"},
}

MEASURES: dict[str, type] = {}  # populated by measures.py via register_measure()

EDGES: dict[str, dict] = {
    "image_in_repository":   {"from": "Image",    "to": "Repository", "table": "image_in_repository",   "join_on": "image_id"},
    "workload_runs_image":   {"from": "Workload",  "to": "Image",      "table": "workload_runs_image_daily", "join_on": "image_id"},
    "workload_in_namespace": {"from": "Workload",  "to": "Namespace",  "table": "workload_in_namespace", "join_on": ("cluster_name", "namespace_name")},
    "namespace_in_cluster":  {"from": "Namespace", "to": "Cluster",    "table": "namespace_in_cluster",  "join_on": ("cluster_name", "namespace_name")},
    "workload_owned_by":     {"from": "Workload",  "to": "Team",       "table": "workload_owned_by",     "join_on": "team_id"},
}


def register_measure(name: str, cls: type) -> None:
    MEASURES[name] = cls
