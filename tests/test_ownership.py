import json
import pytest

from sas.ingest.ownership import (
    LabelStrategy,
    MappingFileStrategy,
    NamespaceFallback,
    ResolverChain,
    OwnershipResult,
)


def test_label_strategy_reads_team_from_namespace_labels():
    labels_json = json.dumps({
        "kubernetes.namespace.label.team": "checkout",
    })
    strat = LabelStrategy(label="team")
    r = strat.resolve(
        cluster="c", namespace="ns", workload_type="Deployment",
        workload_name="w", namespace_labels_json=labels_json,
        agent_tags_json="{}", container_labels_json="{}",
    )
    assert r == OwnershipResult(
        team_id="checkout", owner_id=None,
        resolved_by_strategy="label:team", resolved_from="namespace_labels:team",
    )


def test_label_strategy_returns_none_if_label_absent():
    strat = LabelStrategy(label="team")
    r = strat.resolve(
        cluster="c", namespace="ns", workload_type="Deployment",
        workload_name="w", namespace_labels_json="{}",
        agent_tags_json="{}", container_labels_json="{}",
    )
    assert r is None


def test_mapping_file_strategy_glob_match(fixtures_dir):
    strat = MappingFileStrategy(path=fixtures_dir / "ownership_sample.csv")
    r = strat.resolve(
        cluster="eks-corporate", namespace="platform-a",
        workload_type="Deployment", workload_name="foo",
        namespace_labels_json="{}", agent_tags_json="{}", container_labels_json="{}",
    )
    assert r.team_id == "platform"
    assert r.owner_id == "aaron.miles"


def test_mapping_file_strategy_workload_name_wildcard(fixtures_dir):
    strat = MappingFileStrategy(path=fixtures_dir / "ownership_sample.csv")
    r = strat.resolve(
        cluster="any-cluster", namespace="any-ns",
        workload_type="Deployment", workload_name="audit-service",
        namespace_labels_json="{}", agent_tags_json="{}", container_labels_json="{}",
    )
    assert r.team_id == "security"


def test_namespace_fallback_always_returns_namespace_as_team():
    strat = NamespaceFallback()
    r = strat.resolve(
        cluster="c", namespace="ns-foo",
        workload_type="Deployment", workload_name="w",
        namespace_labels_json="{}", agent_tags_json="{}", container_labels_json="{}",
    )
    assert r == OwnershipResult(
        team_id="ns-foo", owner_id=None,
        resolved_by_strategy="namespace_fallback", resolved_from="namespace:ns-foo",
    )


def test_resolver_chain_first_non_none_wins(fixtures_dir):
    chain = ResolverChain([
        LabelStrategy(label="team"),
        MappingFileStrategy(path=fixtures_dir / "ownership_sample.csv"),
        NamespaceFallback(),
    ])
    # No label, no mapping hit → fallback
    r = chain.resolve(
        cluster="unknown", namespace="random",
        workload_type="Deployment", workload_name="x",
        namespace_labels_json="{}", agent_tags_json="{}", container_labels_json="{}",
    )
    assert r.resolved_by_strategy == "namespace_fallback"
    assert r.team_id == "random"


def test_resolver_chain_label_wins_over_mapping(fixtures_dir):
    chain = ResolverChain([
        LabelStrategy(label="team"),
        MappingFileStrategy(path=fixtures_dir / "ownership_sample.csv"),
        NamespaceFallback(),
    ])
    labels_json = json.dumps({"kubernetes.namespace.label.team": "override"})
    r = chain.resolve(
        cluster="eks-corporate", namespace="platform-a",
        workload_type="Deployment", workload_name="foo",
        namespace_labels_json=labels_json, agent_tags_json="{}", container_labels_json="{}",
    )
    assert r.team_id == "override"
    assert r.resolved_by_strategy == "label:team"
