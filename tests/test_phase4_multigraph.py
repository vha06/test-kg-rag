import os
import sys
import tempfile
import gc
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from storage.db import DatabaseManager
from core.schema import (
    CanonicalDocument,
    ContentBlock,
    CanonicalChunk,
    CanonicalEntity,
    EntityMention,
    CanonicalRelationship,
    EvidenceRecord,
)
from knowledge_graph.multi_graph_builder import MultiDomainGraphBuilder


@pytest.fixture
def temp_db():
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    db = DatabaseManager(db_path=path)
    yield db
    del db
    gc.collect()
    try:
        if os.path.exists(path):
            os.remove(path)
    except PermissionError:
        pass


def test_01_database_platform_tables_creation(temp_db):
    counts = temp_db.count_summary()
    assert "platform_domains" in counts
    assert "platform_entities" in counts
    assert "platform_relationships" in counts
    assert "platform_evidences" in counts
    assert counts["platform_entities"] == 0


def test_02_namespace_isolation(temp_db):
    builder = MultiDomainGraphBuilder(temp_db)

    # Prepare evidence records
    ev_a = EvidenceRecord(
        evidence_id="ev_a1",
        domain_id="domain_alpha",
        document_id="doc_a1",
        quote_text="Alpha server handles routing.",
        verification_status="VERIFIED",
    )
    ev_b = EvidenceRecord(
        evidence_id="ev_b1",
        domain_id="domain_beta",
        document_id="doc_b1",
        quote_text="Beta protocol verifies signatures.",
        verification_status="VERIFIED",
    )
    temp_db.insert_platform_evidences([ev_a, ev_b])

    # Domain Alpha entities & relationship
    ent_a1 = CanonicalEntity(
        entity_id="alpha::ENT::SERVER::gateway",
        canonical_name="Alpha Gateway",
        entity_type="SERVER",
        domain_id="domain_alpha",
        graph_namespace="ns_alpha",
        verification_status="VERIFIED",
        evidence_refs=["ev_a1"],
    )
    ent_a2 = CanonicalEntity(
        entity_id="alpha::ENT::SERVICE::router",
        canonical_name="Router Service",
        entity_type="SERVICE",
        domain_id="domain_alpha",
        graph_namespace="ns_alpha",
        verification_status="VERIFIED",
        evidence_refs=["ev_a1"],
    )
    rel_a = CanonicalRelationship(
        relationship_id="rel_a1",
        subject_entity_id=ent_a1.entity_id,
        predicate="ROUTES_TO",
        object_entity_id=ent_a2.entity_id,
        domain_id="domain_alpha",
        graph_namespace="ns_alpha",
        verification_status="VERIFIED",
        evidence_refs=["ev_a1"],
    )

    # Domain Beta entities & relationship
    ent_b1 = CanonicalEntity(
        entity_id="beta::ENT::PROTOCOL::sig_v1",
        canonical_name="Signature Protocol",
        entity_type="PROTOCOL",
        domain_id="domain_beta",
        graph_namespace="ns_beta",
        verification_status="VERIFIED",
        evidence_refs=["ev_b1"],
    )
    ent_b2 = CanonicalEntity(
        entity_id="beta::ENT::MODULE::crypto",
        canonical_name="Crypto Module",
        entity_type="MODULE",
        domain_id="domain_beta",
        graph_namespace="ns_beta",
        verification_status="VERIFIED",
        evidence_refs=["ev_b1"],
    )
    rel_b = CanonicalRelationship(
        relationship_id="rel_b1",
        subject_entity_id=ent_b1.entity_id,
        predicate="IMPLEMENTED_IN",
        object_entity_id=ent_b2.entity_id,
        domain_id="domain_beta",
        graph_namespace="ns_beta",
        verification_status="VERIFIED",
        evidence_refs=["ev_b1"],
    )

    # Insert all into shared DB
    temp_db.insert_platform_entities([ent_a1, ent_a2, ent_b1, ent_b2])
    temp_db.insert_platform_relationships([rel_a, rel_b])

    # Build Alpha Graph
    graph_a = builder.build_namespace_graph("ns_alpha")
    assert graph_a.number_of_nodes() == 2
    assert graph_a.number_of_edges() == 1
    assert graph_a.has_node("alpha::ENT::SERVER::gateway")
    assert graph_a.has_node("alpha::ENT::SERVICE::router")
    assert not graph_a.has_node("beta::ENT::PROTOCOL::sig_v1")
    assert not graph_a.has_node("beta::ENT::MODULE::crypto")

    # Build Beta Graph
    graph_b = builder.build_namespace_graph("ns_beta")
    assert graph_b.number_of_nodes() == 2
    assert graph_b.number_of_edges() == 1
    assert graph_b.has_node("beta::ENT::PROTOCOL::sig_v1")
    assert not graph_b.has_node("alpha::ENT::SERVER::gateway")


def test_03_candidate_edge_filtering(temp_db):
    builder = MultiDomainGraphBuilder(temp_db)

    ev = EvidenceRecord(
        evidence_id="ev_1",
        domain_id="domain_ops",
        document_id="doc_1",
        quote_text="Quote",
        verification_status="VERIFIED",
    )
    temp_db.insert_platform_evidences([ev])

    ent1 = CanonicalEntity(
        entity_id="ops::ENT::NODE::n1",
        canonical_name="Node 1",
        entity_type="NODE",
        domain_id="domain_ops",
        graph_namespace="ns_ops",
        verification_status="VERIFIED",
        evidence_refs=["ev_1"],
    )
    ent2 = CanonicalEntity(
        entity_id="ops::ENT::NODE::n2",
        canonical_name="Node 2",
        entity_type="NODE",
        domain_id="domain_ops",
        graph_namespace="ns_ops",
        verification_status="VERIFIED",
        evidence_refs=["ev_1"],
    )
    rel_verified = CanonicalRelationship(
        relationship_id="rel_v1",
        subject_entity_id=ent1.entity_id,
        predicate="CONNECTS",
        object_entity_id=ent2.entity_id,
        domain_id="domain_ops",
        graph_namespace="ns_ops",
        verification_status="VERIFIED",
        evidence_refs=["ev_1"],
    )
    rel_unverified = CanonicalRelationship(
        relationship_id="rel_uv1",
        subject_entity_id=ent1.entity_id,
        predicate="SUSPECTED_DEPENDENCY",
        object_entity_id=ent2.entity_id,
        domain_id="domain_ops",
        graph_namespace="ns_ops",
        verification_status="UNVERIFIED",
        confidence=0.4,
        evidence_refs=[],
    )

    temp_db.insert_platform_entities([ent1, ent2])
    temp_db.insert_platform_relationships([rel_verified, rel_unverified])

    # By default: candidates excluded
    graph_default = builder.build_namespace_graph("ns_ops", include_candidates=False)
    assert graph_default.number_of_edges() == 1
    edges_v = list(graph_default.edges(data=True))
    assert edges_v[0][2]["predicate"] == "CONNECTS"

    # With candidates included
    graph_all = builder.build_namespace_graph("ns_ops", include_candidates=True)
    assert graph_all.number_of_edges() == 2


def test_04_provenance_retrieval(temp_db):
    builder = MultiDomainGraphBuilder(temp_db)

    ev = EvidenceRecord(
        evidence_id="ev_audit_100",
        domain_id="domain_ops",
        document_id="doc_ops_1",
        quote_text="Kubernetes ingress controller handles SSL termination.",
        source_location={"start_offset": 50, "end_offset": 105},
        extractor_id="rule_extractor_v1",
        extractor_version="1.0.0",
        verification_status="VERIFIED",
        origin_type="EXTRACTOR_GENERATED",
    )
    temp_db.insert_platform_evidences([ev])

    ent = CanonicalEntity(
        entity_id="ops::ENT::SYSTEM::ingress",
        canonical_name="Ingress Controller",
        entity_type="SYSTEM",
        domain_id="domain_ops",
        graph_namespace="ns_ops",
        verification_status="VERIFIED",
        evidence_refs=["ev_audit_100"],
    )
    ent_target = CanonicalEntity(
        entity_id="ops::ENT::SECURITY::ssl",
        canonical_name="SSL Termination",
        entity_type="SECURITY",
        domain_id="domain_ops",
        graph_namespace="ns_ops",
        verification_status="VERIFIED",
        evidence_refs=["ev_audit_100"],
    )
    rel = CanonicalRelationship(
        relationship_id="rel_ingress_ssl",
        subject_entity_id=ent.entity_id,
        predicate="TERMINATES",
        object_entity_id=ent_target.entity_id,
        domain_id="domain_ops",
        graph_namespace="ns_ops",
        verification_status="VERIFIED",
        evidence_refs=["ev_audit_100"],
    )

    temp_db.insert_platform_entities([ent, ent_target])
    temp_db.insert_platform_relationships([rel])

    node_ev = builder.get_node_provenance(ent.entity_id)
    assert len(node_ev) == 1
    assert node_ev[0]["evidence_id"] == "ev_audit_100"
    assert "Kubernetes ingress controller" in node_ev[0]["quote_text"]

    edge_ev = builder.get_edge_provenance(rel.relationship_id)
    assert len(edge_ev) == 1
    assert edge_ev[0]["evidence_id"] == "ev_audit_100"


def test_05_multi_namespace_and_cross_domain_links(temp_db):
    builder = MultiDomainGraphBuilder(temp_db)

    ev_a = EvidenceRecord(
        evidence_id="ev_a",
        domain_id="dom_a",
        document_id="doc_a",
        quote_text="Quote A",
        verification_status="VERIFIED",
    )
    ev_b = EvidenceRecord(
        evidence_id="ev_b",
        domain_id="dom_b",
        document_id="doc_b",
        quote_text="Quote B",
        verification_status="VERIFIED",
    )
    temp_db.insert_platform_evidences([ev_a, ev_b])

    ent_a = CanonicalEntity(
        entity_id="a::ENT::1",
        canonical_name="Entity A",
        entity_type="TYPE_A",
        domain_id="dom_a",
        graph_namespace="ns_a",
        verification_status="VERIFIED",
        evidence_refs=["ev_a"],
    )
    ent_b = CanonicalEntity(
        entity_id="b::ENT::2",
        canonical_name="Entity B",
        entity_type="TYPE_B",
        domain_id="dom_b",
        graph_namespace="ns_b",
        verification_status="VERIFIED",
        evidence_refs=["ev_b"],
    )
    temp_db.insert_platform_entities([ent_a, ent_b])

    # Insert cross-domain mapping
    mapping = {
        "mapping_id": "map_a1_b2",
        "source_entity_id": ent_a.entity_id,
        "target_entity_id": ent_b.entity_id,
        "source_namespace": "ns_a",
        "target_namespace": "ns_b",
        "relation_type": "EQUIVALENT_TO",
        "verification_status": "VERIFIED",
        "confidence": 0.95,
        "metadata": {"algorithm": "exact_synonym"},
    }
    temp_db.insert_platform_entity_mappings([mapping])

    # Combined without cross-domain links
    g_multi = builder.build_multi_namespace_graph(["ns_a", "ns_b"], include_cross_domain_links=False)
    assert g_multi.number_of_nodes() == 2
    assert g_multi.number_of_edges() == 0

    # Combined with cross-domain links
    g_linked = builder.build_multi_namespace_graph(["ns_a", "ns_b"], include_cross_domain_links=True)
    assert g_linked.number_of_nodes() == 2
    assert g_linked.number_of_edges() == 1
    edge_data = list(g_linked.edges(data=True))[0][2]
    assert edge_data["predicate"] == "EQUIVALENT_TO"
    assert edge_data.get("is_cross_domain") is True
