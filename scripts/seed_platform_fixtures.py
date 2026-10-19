from __future__ import annotations

import os
import sys
import json
import logging
from typing import Dict, Any, List

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
    ACLPolicy,
)
from core.registry import DomainRegistry, ProfileRegistry
from pipeline.orchestrator import PipelineOrchestrator
from pipeline.parsers import ParserRegistry
from pipeline.chunkers import ChunkerRegistry
from pipeline.extractors import ExtractorRegistry
from embedding.adapter import DeterministicHashEmbeddingAdapter
from retrieval.multi_domain_retriever import MultiDomainIndexManager

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def seed_fixtures(
    fixture_path: str = "data/fixtures/synthetic_multi_domain_fixtures.json",
    profiles_yaml_path: str = "config/platform_profiles.yaml",
    db_path: str = "artifacts/legal_lab.db",
) -> Dict[str, Any]:
    """
    Seeds synthetic multi-domain test fixtures into the platform database without
    affecting any legacy legal tables or data.
    """
    logger.info(f"Loading fixture from {fixture_path}...")
    if not os.path.exists(fixture_path):
        raise FileNotFoundError(f"Fixture not found: {fixture_path}")

    with open(fixture_path, "r", encoding="utf-8") as f:
        fixture_data = json.load(f)

    db = DatabaseManager(db_path=db_path)

    # 1. Register Domains and Profiles
    domain_reg = DomainRegistry()
    profile_reg = ProfileRegistry(domain_reg)
    if os.path.exists(profiles_yaml_path):
        domain_reg.load_from_yaml(profiles_yaml_path)
        profile_reg.load_from_yaml(profiles_yaml_path)
        logger.info(f"Loaded {len(domain_reg.list_domains())} domains from {profiles_yaml_path}")
        db.insert_platform_domains(domain_reg.list_domains(enabled_only=False))
        db.insert_platform_profiles(list(profile_reg._profiles.values()))

    orchestrator = PipelineOrchestrator(
        domain_registry=domain_reg,
        profile_registry=profile_reg,
        db_manager=db,
    )

    seeded_docs = 0
    seeded_entities = 0
    seeded_relationships = 0
    seeded_evidences = 0

    # 2. Ingest Domains
    for dom_entry in fixture_data.get("domains", []):
        domain_id = dom_entry["domain_id"]
        graph_ns = dom_entry["graph_namespace"]

        # Insert Evidences first
        ev_objs = []
        for ev in dom_entry.get("evidences", []):
            ev_objs.append(EvidenceRecord.model_validate(ev))
        if ev_objs:
            db.insert_platform_evidences(ev_objs)
            seeded_evidences += len(ev_objs)

        # Ingest Documents through Pipeline Orchestrator
        for doc_entry in dom_entry.get("documents", []):
            raw_text = doc_entry.get("raw_text", "")
            raw_acl = doc_entry.get("acl", {})
            acl_policy = ACLPolicy.model_validate(raw_acl) if raw_acl else ACLPolicy()

            doc_meta = {
                "source_id": doc_entry.get("source_id", "SRC"),
                "domain_id": domain_id,
                "document_type": doc_entry.get("document_type", "generic"),
                "profile_id": doc_entry.get("profile_id", "profile_generic_default"),
                "security_classification": doc_entry.get("security_classification", "INTERNAL"),
                "acl": acl_policy,
            }

            bundle, trace = orchestrator.process_raw_text(
                raw_text=raw_text,
                title=doc_entry["title"],
                document_meta=doc_meta,
                document_id=doc_entry["document_id"],
            )

            # Persist parsed document, blocks, chunks
            db.insert_platform_documents([bundle["document"]])
            db.insert_platform_content_blocks(bundle["blocks"])
            db.insert_platform_chunks(bundle["chunks"])
            seeded_docs += 1

        # Insert Pre-Verified Entities & Relationships for benchmark ground truth
        ent_objs = []
        for ent in dom_entry.get("entities", []):
            ent_objs.append(CanonicalEntity.model_validate(ent))
        if ent_objs:
            db.insert_platform_entities(ent_objs)
            seeded_entities += len(ent_objs)

        rel_objs = []
        for rel in dom_entry.get("relationships", []):
            rel_objs.append(CanonicalRelationship.model_validate(rel))
        if rel_objs:
            db.insert_platform_relationships(rel_objs)
            seeded_relationships += len(rel_objs)

        # Link Entity Mentions to Chunks
        chunks = db.get_platform_chunks(domain_id=domain_id)
        if chunks and ent_objs:
            mentions = []
            for c in chunks:
                for ent in ent_objs:
                    if ent.canonical_name.lower() in c["content"].lower():
                        mentions.append(
                            EntityMention(
                                mention_id=f"men_{c['chunk_id']}_{ent.entity_id}",
                                entity_id=ent.entity_id,
                                domain_id=domain_id,
                                document_id=c["document_id"],
                                chunk_id=c["chunk_id"],
                                surface_text=ent.canonical_name,
                                verification_status="VERIFIED",
                            )
                        )
            if mentions:
                db.insert_platform_entity_mentions(mentions)

    # 3. Insert Cross-Domain Mappings
    mappings = fixture_data.get("cross_domain_mappings", [])
    if mappings:
        db.insert_platform_entity_mappings(mappings)

    # 4. Build Multi-Domain Vector & Lexical Indices
    embedding_adapter = DeterministicHashEmbeddingAdapter(dimension=64)
    idx_mgr = MultiDomainIndexManager(db)
    for dom_entry in fixture_data.get("domains", []):
        d_id = dom_entry["domain_id"]
        idx_mgr.build_index(domain_id=d_id, embedding_adapter=embedding_adapter)

    summary = {
        "status": "SUCCESS",
        "fixture_notice": fixture_data.get("_fixture_notice"),
        "seeded_documents": seeded_docs,
        "seeded_entities": seeded_entities,
        "seeded_relationships": seeded_relationships,
        "seeded_evidences": seeded_evidences,
        "cross_domain_mappings": len(mappings),
        "db_summary": db.count_summary(),
    }
    logger.info(f"Seeding completed successfully: {json.dumps(summary, indent=2)}")
    return summary


if __name__ == "__main__":
    seed_fixtures()
