import os
import sys
import tempfile
import gc
import pytest
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from storage.db import DatabaseManager
from embedding.adapter import DeterministicHashEmbeddingAdapter
from core.schema import (
    CanonicalDocument,
    ContentBlock,
    CanonicalChunk,
    CanonicalEntity,
    EntityMention,
    CanonicalRelationship,
    EvidenceRecord,
    ACLPolicy,
)
from retrieval.multi_domain_retriever import (
    MultiDomainIndexManager,
    MultiDomainHybridRetriever,
    CanonicalRetrievalResult,
)
from knowledge_graph.multi_graph_builder import MultiDomainGraphBuilder
from knowledge_graph.namespace_expander import NamespaceGraphExpander


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


def test_01_deterministic_hash_adapter():
    adapter = DeterministicHashEmbeddingAdapter(dimension=32)
    assert adapter.get_dimension() == 32
    info = adapter.get_model_info()
    assert info["adapter_type"] == "DeterministicHashEmbeddingAdapter"

    texts = ["Server gateway routing", "Cryptographic signature module"]
    vecs = adapter.encode(texts)
    assert vecs.shape == (2, 32)
    assert vecs.dtype == np.float32

    # Determinism: same text gives identical vector
    vecs2 = adapter.encode(texts)
    assert np.allclose(vecs, vecs2)

    # Empty list handling
    empty_vec = adapter.encode([])
    assert empty_vec.shape == (0, 32)


def test_02_multi_domain_hybrid_retriever_and_domain_filtering(temp_db):
    adapter = DeterministicHashEmbeddingAdapter(dimension=32)
    idx_mgr = MultiDomainIndexManager(temp_db)
    retriever = MultiDomainHybridRetriever(temp_db, idx_mgr, adapter)

    # Insert documents
    doc_ops = CanonicalDocument(
        document_id="doc_ops_1",
        title="Ops Manual",
        source_id="ops_1",
        domain_id="domain_ops",
        document_type="manual",
        profile_id="prof_ops",
        checksum="hash_ops",
    )
    doc_rd = CanonicalDocument(
        document_id="doc_rd_1",
        title="RD Spec",
        source_id="rd_1",
        domain_id="domain_rd",
        document_type="spec",
        profile_id="prof_rd",
        checksum="hash_rd",
    )
    temp_db.insert_platform_documents([doc_ops, doc_rd])

    # Insert Chunks
    chunk_ops = CanonicalChunk(
        chunk_id="chk_ops_1",
        document_id="doc_ops_1",
        domain_id="domain_ops",
        source_block_ids=["blk_1"],
        content="The Kubernetes ingress gateway routes HTTP traffic to microservices.",
        chunk_index=0,
        chunking_strategy="sliding_window",
        profile_id="prof_ops",
        security_classification="INTERNAL",
        acl=ACLPolicy(allowed_roles=["ops", "public"]),
    )
    chunk_rd = CanonicalChunk(
        chunk_id="chk_rd_1",
        document_id="doc_rd_1",
        domain_id="domain_rd",
        source_block_ids=["blk_2"],
        content="The neural architecture optimizes attention weights across multi-head layers.",
        chunk_index=0,
        chunking_strategy="sliding_window",
        profile_id="prof_rd",
        security_classification="INTERNAL",
        acl=ACLPolicy(allowed_roles=["researcher", "public"]),
    )
    temp_db.insert_platform_chunks([chunk_ops, chunk_rd])

    # Build indices
    idx_mgr.build_index("domain_ops", embedding_adapter=adapter)
    idx_mgr.build_index("domain_rd", embedding_adapter=adapter)

    # Query only domain_ops
    results_ops = retriever.retrieve(
        query="Kubernetes ingress routing",
        domain_ids=["domain_ops"],
        top_k=5,
    )
    assert len(results_ops) == 1
    assert results_ops[0].domain_id == "domain_ops"
    assert results_ops[0].chunk_id == "chk_ops_1"

    # Query only domain_rd
    results_rd = retriever.retrieve(
        query="attention weights",
        domain_ids=["domain_rd"],
        top_k=5,
    )
    assert len(results_rd) == 1
    assert results_rd[0].domain_id == "domain_rd"
    assert results_rd[0].chunk_id == "chk_rd_1"


def test_03_security_classification_and_acl_enforcement(temp_db):
    adapter = DeterministicHashEmbeddingAdapter(dimension=32)
    idx_mgr = MultiDomainIndexManager(temp_db)
    retriever = MultiDomainHybridRetriever(temp_db, idx_mgr, adapter)

    doc = CanonicalDocument(
        document_id="doc_sec_1",
        title="Sec Policy",
        source_id="sec_1",
        domain_id="domain_sec",
        document_type="policy",
        profile_id="prof_sec",
        checksum="hash_sec",
    )
    temp_db.insert_platform_documents([doc])

    c_public = CanonicalChunk(
        chunk_id="chk_pub",
        document_id="doc_sec_1",
        domain_id="domain_sec",
        source_block_ids=["b1"],
        content="Public security guidelines and FAQs.",
        chunk_index=0,
        chunking_strategy="generic",
        profile_id="prof_sec",
        security_classification="PUBLIC",
        acl=ACLPolicy(allowed_roles=["public"]),
    )
    c_confidential = CanonicalChunk(
        chunk_id="chk_conf",
        document_id="doc_sec_1",
        domain_id="domain_sec",
        source_block_ids=["b2"],
        content="Confidential production encryption root keys and rotation policies.",
        chunk_index=1,
        chunking_strategy="generic",
        profile_id="prof_sec",
        security_classification="CONFIDENTIAL",
        acl=ACLPolicy(allowed_roles=["sec_admin"]),
    )
    temp_db.insert_platform_chunks([c_public, c_confidential])
    idx_mgr.build_index("domain_sec", embedding_adapter=adapter)

    # 1. User with only PUBLIC classification access
    pub_results = retriever.retrieve(
        query="security encryption keys",
        domain_ids=["domain_sec"],
        allowed_classifications=["PUBLIC"],
        user_roles=["public"],
    )
    assert len(pub_results) == 1
    assert pub_results[0].chunk_id == "chk_pub"

    # 2. User with CONFIDENTIAL access but missing "sec_admin" role
    denied_role_results = retriever.retrieve(
        query="security encryption keys",
        domain_ids=["domain_sec"],
        allowed_classifications=["PUBLIC", "CONFIDENTIAL"],
        user_roles=["general_dev"],
    )
    # The confidential chunk requires role "sec_admin", so only public chunk is returned
    assert len(denied_role_results) == 1
    assert denied_role_results[0].chunk_id == "chk_pub"

    # 3. User with CONFIDENTIAL access AND "sec_admin" role
    authorized_results = retriever.retrieve(
        query="security encryption keys",
        domain_ids=["domain_sec"],
        allowed_classifications=["PUBLIC", "CONFIDENTIAL"],
        user_roles=["sec_admin"],
    )
    assert len(authorized_results) == 2


def test_04_namespace_graph_expander(temp_db):
    adapter = DeterministicHashEmbeddingAdapter(dimension=32)
    graph_builder = MultiDomainGraphBuilder(temp_db)
    expander = NamespaceGraphExpander(temp_db, graph_builder)

    # Setup Evidence & Entities
    ev = EvidenceRecord(
        evidence_id="ev_exp_1",
        domain_id="domain_ops",
        document_id="doc_ops_1",
        quote_text="Ingress proxies to Backend service",
        verification_status="VERIFIED",
    )
    temp_db.insert_platform_evidences([ev])

    ent_ingress = CanonicalEntity(
        entity_id="ops::ENT::SYS::ingress",
        canonical_name="Ingress",
        entity_type="SYSTEM",
        domain_id="domain_ops",
        graph_namespace="ns_ops",
        verification_status="VERIFIED",
        evidence_refs=["ev_exp_1"],
    )
    ent_backend = CanonicalEntity(
        entity_id="ops::ENT::SYS::backend",
        canonical_name="Backend Service",
        entity_type="SYSTEM",
        domain_id="domain_ops",
        graph_namespace="ns_ops",
        verification_status="VERIFIED",
        evidence_refs=["ev_exp_1"],
    )
    rel = CanonicalRelationship(
        relationship_id="rel_ing_back",
        subject_entity_id=ent_ingress.entity_id,
        predicate="PROXIES_TO",
        object_entity_id=ent_backend.entity_id,
        domain_id="domain_ops",
        graph_namespace="ns_ops",
        verification_status="VERIFIED",
        evidence_refs=["ev_exp_1"],
    )
    temp_db.insert_platform_entities([ent_ingress, ent_backend])
    temp_db.insert_platform_relationships([rel])

    # Mention linking ingress to chunk_1
    mention = EntityMention(
        mention_id="men_1",
        entity_id=ent_ingress.entity_id,
        domain_id="domain_ops",
        document_id="doc_ops_1",
        chunk_id="chunk_ops_seed",
        surface_text="Ingress",
        verification_status="VERIFIED",
    )
    temp_db.insert_platform_entity_mentions([mention])

    # Seed result
    seed = CanonicalRetrievalResult(
        chunk_id="chunk_ops_seed",
        document_id="doc_ops_1",
        domain_id="domain_ops",
        content="Ingress receives external traffic.",
        score=0.9,
    )

    # Expand 1 hop within ns_ops
    expansion = expander.expand(
        seed_results=[seed],
        max_hops=1,
        allowed_namespaces=["ns_ops"],
    )

    assert "chunk_ops_seed" in expansion["seed_chunk_ids"]
    assert ent_ingress.entity_id in expansion["seed_entity_ids"]
    assert len(expansion["expanded_entities"]) == 1
    expanded_node = expansion["expanded_entities"][0]
    assert expanded_node["entity_id"] == ent_backend.entity_id
    assert expanded_node["canonical_name"] == "Backend Service"
    assert len(expanded_node["provenance"]) == 1
    assert expanded_node["provenance"][0]["evidence_id"] == "ev_exp_1"
