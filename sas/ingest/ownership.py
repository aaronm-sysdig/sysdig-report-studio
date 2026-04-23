"""Ownership resolver chain. Returns (team_id, owner_id) for a workload.

Strategies are evaluated in order; first non-None wins. Every result carries
resolved_by_strategy + resolved_from for auditability.
"""
from __future__ import annotations

import csv
import fnmatch
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Protocol


@dataclass(frozen=True)
class OwnershipResult:
    team_id: Optional[str]
    owner_id: Optional[str]
    resolved_by_strategy: str
    resolved_from: str


class Strategy(Protocol):
    def resolve(self, *, cluster: str, namespace: str, workload_type: str,
                workload_name: str, namespace_labels_json: str,
                agent_tags_json: str, container_labels_json: str
                ) -> Optional[OwnershipResult]: ...


_LABEL_PREFIXES = [
    "kubernetes.namespace.label.",
    "kube.label.",
    "",  # raw key, in case the label is stored ungilded
]


class LabelStrategy:
    """Look for a configured label in namespace_labels / agent_tags / container_labels."""

    def __init__(self, label: str):
        self.label = label

    def resolve(self, *, cluster, namespace, workload_type, workload_name,
                namespace_labels_json, agent_tags_json, container_labels_json):
        for source_name, blob in (
            ("namespace_labels", namespace_labels_json),
            ("agent_tags", agent_tags_json),
            ("container_labels", container_labels_json),
        ):
            try:
                d = json.loads(blob) if blob else {}
            except (json.JSONDecodeError, TypeError):
                continue
            if not isinstance(d, dict):
                continue
            for prefix in _LABEL_PREFIXES:
                key = f"{prefix}{self.label}"
                if key in d and d[key]:
                    return OwnershipResult(
                        team_id=str(d[key]),
                        owner_id=None,
                        resolved_by_strategy=f"label:{self.label}",
                        resolved_from=f"{source_name}:{self.label}",
                    )
        return None


class MappingFileStrategy:
    """CSV with columns: cluster,namespace,workload_type,workload_name,team,owner.

    Values accept glob wildcards ('*'). First matching row wins. File is re-read
    on each resolve call — acceptable for ingest-time use (not per-query hot path).
    """

    def __init__(self, path: Path):
        self.path = path

    def resolve(self, *, cluster, namespace, workload_type, workload_name,
                namespace_labels_json, agent_tags_json, container_labels_json):
        if not self.path.exists():
            return None
        with open(self.path, "r", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if (fnmatch.fnmatchcase(cluster, row.get("cluster", "*") or "*")
                    and fnmatch.fnmatchcase(namespace, row.get("namespace", "*") or "*")
                    and fnmatch.fnmatchcase(workload_type, row.get("workload_type", "*") or "*")
                    and fnmatch.fnmatchcase(workload_name, row.get("workload_name", "*") or "*")):
                    team = row.get("team") or None
                    owner = row.get("owner") or None
                    return OwnershipResult(
                        team_id=team, owner_id=owner,
                        resolved_by_strategy="mapping_file",
                        resolved_from=f"{self.path.name}:row_match",
                    )
        return None


class NamespaceFallback:
    """Last-resort: team = namespace, no owner."""

    def resolve(self, *, cluster, namespace, workload_type, workload_name,
                namespace_labels_json, agent_tags_json, container_labels_json):
        return OwnershipResult(
            team_id=namespace,
            owner_id=None,
            resolved_by_strategy="namespace_fallback",
            resolved_from=f"namespace:{namespace}",
        )


class ResolverChain:
    def __init__(self, strategies: list[Strategy]):
        self.strategies = strategies

    def resolve(self, **kwargs) -> OwnershipResult:
        for strat in self.strategies:
            r = strat.resolve(**kwargs)
            if r is not None:
                return r
        # Belt-and-braces: if no fallback was provided, synthesize one.
        return NamespaceFallback().resolve(**kwargs)
