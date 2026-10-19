import os
import sqlite3
import json
from typing import List, Optional, Dict, Any
from storage.models import (
    LegalDocumentModel,
    ProvisionModel,
    ProvisionVersionModel,
    ChunkModel,
    KGEdgeModel,
)


class DatabaseManager:
    def __init__(self, db_path: str = "artifacts/legal_lab.db"):
        self.db_path = db_path
        os.makedirs(os.path.dirname(os.path.abspath(db_path)), exist_ok=True)
        self._init_db()

    def get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        return conn

    def _init_db(self):
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.executescript("""
                CREATE TABLE IF NOT EXISTS documents (
                    document_id TEXT PRIMARY KEY,
                    doc_number TEXT NOT NULL,
                    title TEXT NOT NULL,
                    doc_type TEXT NOT NULL,
                    issuer TEXT NOT NULL,
                    issue_date TEXT,
                    effective_start TEXT,
                    effective_end TEXT,
                    status TEXT DEFAULT 'ACTIVE',
                    source_url TEXT,
                    content_hash TEXT
                );

                CREATE TABLE IF NOT EXISTS provisions (
                    provision_id TEXT PRIMARY KEY,
                    document_id TEXT NOT NULL,
                    parent_id TEXT,
                    level TEXT NOT NULL,
                    article_number TEXT,
                    clause_number TEXT,
                    point_letter TEXT,
                    title TEXT,
                    structural_path TEXT,
                    content TEXT,
                    mapc TEXT,
                    effective_start TEXT,
                    effective_end TEXT,
                    status TEXT DEFAULT 'CÓ_HIỆU_LỰC_THI_HÀNH',
                    source_url TEXT,
                    is_external_stub INTEGER DEFAULT 0,
                    FOREIGN KEY (document_id) REFERENCES documents (document_id) ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS provision_versions (
                    version_id TEXT PRIMARY KEY,
                    provision_id TEXT NOT NULL,
                    version_num INTEGER NOT NULL,
                    effective_start TEXT,
                    effective_end TEXT,
                    status TEXT,
                    amending_provision_id TEXT,
                    content TEXT,
                    version_note TEXT,
                    FOREIGN KEY (provision_id) REFERENCES provisions (provision_id) ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS chunks (
                    chunk_id TEXT PRIMARY KEY,
                    strategy TEXT NOT NULL,
                    document_id TEXT NOT NULL,
                    provision_id TEXT NOT NULL,
                    source_provision_ids TEXT,
                    structural_path TEXT,
                    content TEXT NOT NULL,
                    token_count INTEGER NOT NULL,
                    crosses_boundary INTEGER DEFAULT 0,
                    effective_start TEXT,
                    effective_end TEXT,
                    status TEXT,
                    source_url TEXT,
                    content_hash TEXT,
                    FOREIGN KEY (document_id) REFERENCES documents (document_id) ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS kg_edges (
                    edge_id TEXT PRIMARY KEY,
                    source_id TEXT NOT NULL,
                    target_id TEXT NOT NULL,
                    relation_type TEXT NOT NULL,
                    evidence_text TEXT,
                    source_url TEXT,
                    effective_date TEXT,
                    detection_method TEXT,
                    verified INTEGER DEFAULT 1
                );

                CREATE TABLE IF NOT EXISTS index_manifests (
                    index_id TEXT PRIMARY KEY,
                    strategy TEXT NOT NULL,
                    model_name TEXT NOT NULL,
                    corpus_hash TEXT NOT NULL,
                    chunk_count INTEGER NOT NULL,
                    build_time_sec REAL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    config_json TEXT
                );

                CREATE INDEX IF NOT EXISTS idx_provisions_doc ON provisions (document_id);
                CREATE INDEX IF NOT EXISTS idx_provisions_mapc ON provisions (mapc);
                CREATE INDEX IF NOT EXISTS idx_provisions_level ON provisions (level);
                CREATE INDEX IF NOT EXISTS idx_chunks_strat ON chunks (strategy);
                CREATE INDEX IF NOT EXISTS idx_chunks_prov ON chunks (provision_id);
                CREATE INDEX IF NOT EXISTS idx_kg_src ON kg_edges (source_id);
                CREATE INDEX IF NOT EXISTS idx_kg_tgt ON kg_edges (target_id);
                CREATE INDEX IF NOT EXISTS idx_kg_rel ON kg_edges (relation_type);

                -- Platform Architecture Multi-Domain Tables
                CREATE TABLE IF NOT EXISTS platform_domains (
                    domain_id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    description TEXT,
                    enabled INTEGER DEFAULT 1,
                    schema_version TEXT DEFAULT '1.0.0',
                    graph_namespace TEXT NOT NULL,
                    default_profile_id TEXT,
                    is_test_fixture INTEGER DEFAULT 0,
                    allowed_entity_types TEXT,
                    allowed_predicates TEXT
                );

                CREATE TABLE IF NOT EXISTS platform_profiles (
                    profile_id TEXT PRIMARY KEY,
                    domain_id TEXT NOT NULL,
                    document_type TEXT NOT NULL,
                    profile_version TEXT NOT NULL,
                    parser_strategy TEXT NOT NULL,
                    chunking_strategy TEXT NOT NULL,
                    chunking_parameters TEXT,
                    metadata_extractors TEXT,
                    entity_schema_ref TEXT,
                    relationship_schema_ref TEXT,
                    validation_rules TEXT,
                    fallback_profile_id TEXT,
                    enabled INTEGER DEFAULT 1
                );

                CREATE TABLE IF NOT EXISTS platform_documents (
                    document_id TEXT PRIMARY KEY,
                    schema_version TEXT NOT NULL,
                    title TEXT NOT NULL,
                    source_id TEXT NOT NULL,
                    domain_id TEXT NOT NULL,
                    document_type TEXT NOT NULL,
                    profile_id TEXT NOT NULL,
                    profile_version TEXT NOT NULL,
                    checksum TEXT NOT NULL,
                    ingested_at TEXT NOT NULL,
                    processing_status TEXT NOT NULL,
                    security_classification TEXT NOT NULL,
                    source_uri TEXT,
                    mime_type TEXT,
                    language TEXT,
                    acl TEXT,
                    metadata TEXT
                );

                CREATE TABLE IF NOT EXISTS platform_content_blocks (
                    block_id TEXT PRIMARY KEY,
                    document_id TEXT NOT NULL,
                    domain_id TEXT NOT NULL,
                    block_index INTEGER NOT NULL,
                    block_type TEXT NOT NULL,
                    text TEXT NOT NULL,
                    parent_block_id TEXT,
                    source_location TEXT,
                    start_offset INTEGER,
                    end_offset INTEGER,
                    metadata TEXT,
                    FOREIGN KEY (document_id) REFERENCES platform_documents(document_id) ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS platform_chunks (
                    chunk_id TEXT PRIMARY KEY,
                    document_id TEXT NOT NULL,
                    domain_id TEXT NOT NULL,
                    source_block_ids TEXT NOT NULL,
                    content TEXT NOT NULL,
                    chunk_index INTEGER NOT NULL,
                    chunking_strategy TEXT NOT NULL,
                    profile_id TEXT NOT NULL,
                    profile_version TEXT NOT NULL,
                    token_count INTEGER NOT NULL,
                    source_location TEXT,
                    security_classification TEXT NOT NULL,
                    acl TEXT,
                    metadata TEXT,
                    FOREIGN KEY (document_id) REFERENCES platform_documents(document_id) ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS platform_evidences (
                    evidence_id TEXT PRIMARY KEY,
                    domain_id TEXT NOT NULL,
                    document_id TEXT NOT NULL,
                    block_id TEXT,
                    chunk_id TEXT,
                    quote_text TEXT NOT NULL,
                    source_location TEXT,
                    extractor_id TEXT NOT NULL,
                    extractor_version TEXT NOT NULL,
                    extracted_at TEXT NOT NULL,
                    verification_status TEXT NOT NULL,
                    origin_type TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS platform_entities (
                    entity_id TEXT PRIMARY KEY,
                    canonical_name TEXT NOT NULL,
                    entity_type TEXT NOT NULL,
                    aliases TEXT,
                    description TEXT,
                    domain_id TEXT NOT NULL,
                    graph_namespace TEXT NOT NULL,
                    extraction_method TEXT NOT NULL,
                    extractor_version TEXT NOT NULL,
                    verification_status TEXT NOT NULL,
                    schema_version TEXT NOT NULL,
                    attributes TEXT,
                    evidence_refs TEXT
                );

                CREATE TABLE IF NOT EXISTS platform_entity_mentions (
                    mention_id TEXT PRIMARY KEY,
                    entity_id TEXT NOT NULL,
                    domain_id TEXT NOT NULL,
                    document_id TEXT NOT NULL,
                    block_id TEXT,
                    chunk_id TEXT,
                    surface_text TEXT NOT NULL,
                    source_location TEXT,
                    verification_status TEXT NOT NULL,
                    evidence_id TEXT,
                    FOREIGN KEY (entity_id) REFERENCES platform_entities(entity_id) ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS platform_relationships (
                    relationship_id TEXT PRIMARY KEY,
                    subject_entity_id TEXT NOT NULL,
                    predicate TEXT NOT NULL,
                    object_entity_id TEXT NOT NULL,
                    domain_id TEXT NOT NULL,
                    graph_namespace TEXT NOT NULL,
                    evidence_refs TEXT NOT NULL,
                    extraction_method TEXT NOT NULL,
                    extractor_version TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    verification_status TEXT NOT NULL,
                    recorded_at TEXT NOT NULL,
                    schema_version TEXT NOT NULL,
                    attributes TEXT,
                    FOREIGN KEY (subject_entity_id) REFERENCES platform_entities(entity_id) ON DELETE CASCADE,
                    FOREIGN KEY (object_entity_id) REFERENCES platform_entities(entity_id) ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS platform_entity_mappings (
                    mapping_id TEXT PRIMARY KEY,
                    source_entity_id TEXT NOT NULL,
                    target_entity_id TEXT NOT NULL,
                    source_namespace TEXT NOT NULL,
                    target_namespace TEXT NOT NULL,
                    relation_type TEXT NOT NULL,
                    verification_status TEXT NOT NULL,
                    confidence REAL DEFAULT 1.0,
                    metadata TEXT
                );

                CREATE INDEX IF NOT EXISTS idx_platform_docs_domain ON platform_documents (domain_id);
                CREATE INDEX IF NOT EXISTS idx_platform_blocks_doc ON platform_content_blocks (document_id);
                CREATE INDEX IF NOT EXISTS idx_platform_chunks_domain ON platform_chunks (domain_id);
                CREATE INDEX IF NOT EXISTS idx_platform_chunks_doc ON platform_chunks (document_id);
                CREATE INDEX IF NOT EXISTS idx_platform_entities_ns ON platform_entities (graph_namespace);
                CREATE INDEX IF NOT EXISTS idx_platform_entities_domain ON platform_entities (domain_id);
                CREATE INDEX IF NOT EXISTS idx_platform_rel_ns ON platform_relationships (graph_namespace);
                CREATE INDEX IF NOT EXISTS idx_platform_rel_domain ON platform_relationships (domain_id);
                CREATE INDEX IF NOT EXISTS idx_platform_rel_src ON platform_relationships (subject_entity_id);
                CREATE INDEX IF NOT EXISTS idx_platform_rel_tgt ON platform_relationships (object_entity_id);
                CREATE INDEX IF NOT EXISTS idx_platform_evid_domain ON platform_evidences (domain_id);
                CREATE INDEX IF NOT EXISTS idx_platform_evid_doc ON platform_evidences (document_id);
            """)
            conn.commit()

    def insert_documents(self, docs: List[LegalDocumentModel]):
        with self.get_connection() as conn:
            cursor = conn.cursor()
            for doc in docs:
                cursor.execute("""
                    INSERT OR REPLACE INTO documents 
                    (document_id, doc_number, title, doc_type, issuer, issue_date, effective_start, effective_end, status, source_url, content_hash)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    doc.document_id, doc.doc_number, doc.title, doc.doc_type, doc.issuer,
                    doc.issue_date, doc.effective_start, doc.effective_end, doc.status,
                    doc.source_url, doc.content_hash
                ))
                for prov in doc.provisions:
                    cursor.execute("""
                        INSERT OR REPLACE INTO provisions
                        (provision_id, document_id, parent_id, level, article_number, clause_number, point_letter,
                         title, structural_path, content, mapc, effective_start, effective_end, status, source_url, is_external_stub)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        prov.provision_id, prov.document_id, prov.parent_id, prov.level,
                        prov.article_number, prov.clause_number, prov.point_letter, prov.title,
                        prov.structural_path, prov.content, prov.mapc, prov.effective_start,
                        prov.effective_end, prov.status, prov.source_url, prov.is_external_stub
                    ))
                    for ver in prov.versions:
                        cursor.execute("""
                            INSERT OR REPLACE INTO provision_versions
                            (version_id, provision_id, version_num, effective_start, effective_end, status,
                             amending_provision_id, content, version_note)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """, (
                            ver.version_id, ver.provision_id, ver.version_num, ver.effective_start,
                            ver.effective_end, ver.status, ver.amending_provision_id, ver.content, ver.version_note
                        ))
            conn.commit()

    def insert_chunks(self, chunks: List[ChunkModel]):
        with self.get_connection() as conn:
            cursor = conn.cursor()
            for c in chunks:
                cursor.execute("""
                    INSERT OR REPLACE INTO chunks
                    (chunk_id, strategy, document_id, provision_id, source_provision_ids, structural_path,
                     content, token_count, crosses_boundary, effective_start, effective_end, status, source_url, content_hash)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    c.chunk_id, c.strategy, c.document_id, c.provision_id,
                    json.dumps(c.source_provision_ids, ensure_ascii=False),
                    c.structural_path, c.content, c.token_count, 1 if c.crosses_boundary else 0,
                    c.effective_start, c.effective_end, c.status, c.source_url, c.content_hash
                ))
            conn.commit()

    def insert_kg_edges(self, edges: List[KGEdgeModel]):
        with self.get_connection() as conn:
            cursor = conn.cursor()
            for e in edges:
                cursor.execute("""
                    INSERT OR REPLACE INTO kg_edges
                    (edge_id, source_id, target_id, relation_type, evidence_text, source_url, effective_date, detection_method, verified)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    e.edge_id, e.source_id, e.target_id, e.relation_type,
                    e.evidence_text, e.source_url, e.effective_date, e.detection_method, e.verified
                ))
            conn.commit()

    def get_chunks_by_strategy(self, strategy: str) -> List[Dict[str, Any]]:
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT c.*, d.doc_number, d.title as doc_title, p.title as provision_title
                FROM chunks c
                JOIN documents d ON c.document_id = d.document_id
                JOIN provisions p ON c.provision_id = p.provision_id
                WHERE c.strategy = ?
            """, (strategy,))
            return [dict(row) for row in cursor.fetchall()]

    def get_provision(self, provision_id: str) -> Optional[Dict[str, Any]]:
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT p.*, d.doc_number, d.title as doc_title
                FROM provisions p
                JOIN documents d ON p.document_id = d.document_id
                WHERE p.provision_id = ?
            """, (provision_id,))
            row = cursor.fetchone()
            return dict(row) if row else None

    def get_kg_edges(self, verified_only: bool = True) -> List[Dict[str, Any]]:
        with self.get_connection() as conn:
            cursor = conn.cursor()
            query = "SELECT * FROM kg_edges"
            if verified_only:
                query += " WHERE verified = 1"
            cursor.execute(query)
            return [dict(row) for row in cursor.fetchall()]

    def count_summary(self) -> Dict[str, int]:
        with self.get_connection() as conn:
            cursor = conn.cursor()
            counts = {}
            for table in [
                "documents", "provisions", "provision_versions", "chunks", "kg_edges",
                "platform_domains", "platform_profiles", "platform_documents",
                "platform_content_blocks", "platform_chunks", "platform_evidences",
                "platform_entities", "platform_entity_mentions", "platform_relationships",
                "platform_entity_mappings"
            ]:
                try:
                    cursor.execute(f"SELECT COUNT(*) FROM {table}")
                    counts[table] = cursor.fetchone()[0]
                except sqlite3.OperationalError:
                    counts[table] = 0
            return counts

    # ==========================================
    # PLATFORM ARCHITECTURE CRUD METHODS
    # ==========================================

    @staticmethod
    def _to_dict(obj: Any) -> Dict[str, Any]:
        if hasattr(obj, "model_dump"):
            return obj.model_dump()
        return dict(obj)

    @staticmethod
    def _to_json(val: Any) -> str:
        if val is None:
            return "{}"
        if hasattr(val, "model_dump"):
            val = val.model_dump()
        if isinstance(val, (dict, list)):
            return json.dumps(val, ensure_ascii=False)
        return str(val)

    def insert_platform_domains(self, domains: List[Any]):
        with self.get_connection() as conn:
            cursor = conn.cursor()
            for d in domains:
                data = self._to_dict(d)
                cursor.execute("""
                    INSERT OR REPLACE INTO platform_domains
                    (domain_id, name, description, enabled, schema_version, graph_namespace, default_profile_id, is_test_fixture, allowed_entity_types, allowed_predicates)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    data.get("domain_id"),
                    data.get("name"),
                    data.get("description", ""),
                    1 if data.get("enabled", True) else 0,
                    data.get("schema_version", "1.0.0"),
                    data.get("graph_namespace"),
                    data.get("default_profile_id"),
                    1 if data.get("is_test_fixture", False) else 0,
                    self._to_json(data.get("allowed_entity_types", [])),
                    self._to_json(data.get("allowed_predicates", []))
                ))
            conn.commit()

    def insert_platform_profiles(self, profiles: List[Any]):
        with self.get_connection() as conn:
            cursor = conn.cursor()
            for p in profiles:
                data = self._to_dict(p)
                cursor.execute("""
                    INSERT OR REPLACE INTO platform_profiles
                    (profile_id, domain_id, document_type, profile_version, parser_strategy, chunking_strategy, chunking_parameters, metadata_extractors, entity_schema_ref, relationship_schema_ref, validation_rules, fallback_profile_id, enabled)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    data.get("profile_id"),
                    data.get("domain_id"),
                    data.get("document_type"),
                    data.get("profile_version", "1.0.0"),
                    data.get("parser_strategy"),
                    data.get("chunking_strategy"),
                    self._to_json(data.get("chunking_parameters", {})),
                    self._to_json(data.get("metadata_extractors", [])),
                    data.get("entity_schema_ref"),
                    data.get("relationship_schema_ref"),
                    self._to_json(data.get("validation_rules", {})),
                    data.get("fallback_profile_id"),
                    1 if data.get("enabled", True) else 0
                ))
            conn.commit()

    def insert_platform_documents(self, docs: List[Any]):
        with self.get_connection() as conn:
            cursor = conn.cursor()
            for doc in docs:
                data = self._to_dict(doc)
                cursor.execute("""
                    INSERT OR REPLACE INTO platform_documents
                    (document_id, schema_version, title, source_id, domain_id, document_type, profile_id, profile_version, checksum, ingested_at, processing_status, security_classification, source_uri, mime_type, language, acl, metadata)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    data.get("document_id"),
                    data.get("schema_version", "1.0.0"),
                    data.get("title"),
                    data.get("source_id"),
                    data.get("domain_id"),
                    data.get("document_type"),
                    data.get("profile_id"),
                    data.get("profile_version", "1.0.0"),
                    data.get("checksum"),
                    data.get("ingested_at"),
                    data.get("processing_status", "PENDING"),
                    data.get("security_classification", "INTERNAL"),
                    data.get("source_uri"),
                    data.get("mime_type", "text/plain"),
                    data.get("language", "vi"),
                    self._to_json(data.get("acl", {})),
                    self._to_json(data.get("metadata", {}))
                ))
            conn.commit()

    def insert_platform_content_blocks(self, blocks: List[Any]):
        with self.get_connection() as conn:
            cursor = conn.cursor()
            for b in blocks:
                data = self._to_dict(b)
                cursor.execute("""
                    INSERT OR REPLACE INTO platform_content_blocks
                    (block_id, document_id, domain_id, block_index, block_type, text, parent_block_id, source_location, start_offset, end_offset, metadata)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    data.get("block_id"),
                    data.get("document_id"),
                    data.get("domain_id"),
                    data.get("block_index", 0),
                    data.get("block_type", "PARAGRAPH"),
                    data.get("text", ""),
                    data.get("parent_block_id"),
                    self._to_json(data.get("source_location", {})),
                    data.get("start_offset"),
                    data.get("end_offset"),
                    self._to_json(data.get("metadata", {}))
                ))
            conn.commit()

    def insert_platform_chunks(self, chunks: List[Any]):
        with self.get_connection() as conn:
            cursor = conn.cursor()
            for c in chunks:
                data = self._to_dict(c)
                cursor.execute("""
                    INSERT OR REPLACE INTO platform_chunks
                    (chunk_id, document_id, domain_id, source_block_ids, content, chunk_index, chunking_strategy, profile_id, profile_version, token_count, source_location, security_classification, acl, metadata)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    data.get("chunk_id"),
                    data.get("document_id"),
                    data.get("domain_id"),
                    self._to_json(data.get("source_block_ids", [])),
                    data.get("content", ""),
                    data.get("chunk_index", 0),
                    data.get("chunking_strategy", "generic"),
                    data.get("profile_id"),
                    data.get("profile_version", "1.0.0"),
                    data.get("token_count", 0),
                    self._to_json(data.get("source_location", {})),
                    data.get("security_classification", "INTERNAL"),
                    self._to_json(data.get("acl", {})),
                    self._to_json(data.get("metadata", {}))
                ))
            conn.commit()

    def insert_platform_evidences(self, evidences: List[Any]):
        with self.get_connection() as conn:
            cursor = conn.cursor()
            for ev in evidences:
                data = self._to_dict(ev)
                cursor.execute("""
                    INSERT OR REPLACE INTO platform_evidences
                    (evidence_id, domain_id, document_id, block_id, chunk_id, quote_text, source_location, extractor_id, extractor_version, extracted_at, verification_status, origin_type)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    data.get("evidence_id"),
                    data.get("domain_id"),
                    data.get("document_id"),
                    data.get("block_id"),
                    data.get("chunk_id"),
                    data.get("quote_text", ""),
                    self._to_json(data.get("source_location", {})),
                    data.get("extractor_id", "system"),
                    data.get("extractor_version", "1.0.0"),
                    data.get("extracted_at"),
                    data.get("verification_status", "UNVERIFIED"),
                    data.get("origin_type", "EXTRACTOR_GENERATED")
                ))
            conn.commit()

    def insert_platform_entities(self, entities: List[Any]):
        with self.get_connection() as conn:
            cursor = conn.cursor()
            for ent in entities:
                data = self._to_dict(ent)
                cursor.execute("""
                    INSERT OR REPLACE INTO platform_entities
                    (entity_id, canonical_name, entity_type, aliases, description, domain_id, graph_namespace, extraction_method, extractor_version, verification_status, schema_version, attributes, evidence_refs)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    data.get("entity_id"),
                    data.get("canonical_name"),
                    data.get("entity_type"),
                    self._to_json(data.get("aliases", [])),
                    data.get("description"),
                    data.get("domain_id"),
                    data.get("graph_namespace"),
                    data.get("extraction_method", "RULE_BASED"),
                    data.get("extractor_version", "1.0.0"),
                    data.get("verification_status", "UNVERIFIED"),
                    data.get("schema_version", "1.0.0"),
                    self._to_json(data.get("attributes", {})),
                    self._to_json(data.get("evidence_refs", []))
                ))
            conn.commit()

    def insert_platform_entity_mentions(self, mentions: List[Any]):
        with self.get_connection() as conn:
            cursor = conn.cursor()
            for m in mentions:
                data = self._to_dict(m)
                cursor.execute("""
                    INSERT OR REPLACE INTO platform_entity_mentions
                    (mention_id, entity_id, domain_id, document_id, block_id, chunk_id, surface_text, source_location, verification_status, evidence_id)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    data.get("mention_id"),
                    data.get("entity_id"),
                    data.get("domain_id"),
                    data.get("document_id"),
                    data.get("block_id"),
                    data.get("chunk_id"),
                    data.get("surface_text"),
                    self._to_json(data.get("source_location", {})),
                    data.get("verification_status", "UNVERIFIED"),
                    data.get("evidence_id")
                ))
            conn.commit()

    def insert_platform_relationships(self, relationships: List[Any]):
        with self.get_connection() as conn:
            cursor = conn.cursor()
            for rel in relationships:
                data = self._to_dict(rel)
                cursor.execute("""
                    INSERT OR REPLACE INTO platform_relationships
                    (relationship_id, subject_entity_id, predicate, object_entity_id, domain_id, graph_namespace, evidence_refs, extraction_method, extractor_version, confidence, verification_status, recorded_at, schema_version, attributes)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    data.get("relationship_id"),
                    data.get("subject_entity_id"),
                    data.get("predicate"),
                    data.get("object_entity_id"),
                    data.get("domain_id"),
                    data.get("graph_namespace"),
                    self._to_json(data.get("evidence_refs", [])),
                    data.get("extraction_method", "RULE_BASED"),
                    data.get("extractor_version", "1.0.0"),
                    float(data.get("confidence", 1.0)),
                    data.get("verification_status", "UNVERIFIED"),
                    data.get("recorded_at"),
                    data.get("schema_version", "1.0.0"),
                    self._to_json(data.get("attributes", {}))
                ))
            conn.commit()

    def insert_platform_entity_mappings(self, mappings: List[Dict[str, Any]]):
        with self.get_connection() as conn:
            cursor = conn.cursor()
            for m in mappings:
                cursor.execute("""
                    INSERT OR REPLACE INTO platform_entity_mappings
                    (mapping_id, source_entity_id, target_entity_id, source_namespace, target_namespace, relation_type, verification_status, confidence, metadata)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    m.get("mapping_id"),
                    m.get("source_entity_id"),
                    m.get("target_entity_id"),
                    m.get("source_namespace"),
                    m.get("target_namespace"),
                    m.get("relation_type", "SAME_AS"),
                    m.get("verification_status", "VERIFIED"),
                    float(m.get("confidence", 1.0)),
                    self._to_json(m.get("metadata", {}))
                ))
            conn.commit()

    def insert_platform_bundle(
        self,
        doc: Any,
        blocks: List[Any],
        chunks: List[Any],
        entities: List[Any],
        mentions: List[Any],
        relationships: List[Any],
        evidences: List[Any],
    ):
        """Atomically insert an entire parsed/extracted bundle into the platform schema."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            # 1. Document
            doc_data = self._to_dict(doc)
            cursor.execute("""
                INSERT OR REPLACE INTO platform_documents
                (document_id, schema_version, title, source_id, domain_id, document_type, profile_id, profile_version, checksum, ingested_at, processing_status, security_classification, source_uri, mime_type, language, acl, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                doc_data.get("document_id"), doc_data.get("schema_version", "1.0.0"),
                doc_data.get("title"), doc_data.get("source_id"), doc_data.get("domain_id"),
                doc_data.get("document_type"), doc_data.get("profile_id"), doc_data.get("profile_version", "1.0.0"),
                doc_data.get("checksum"), doc_data.get("ingested_at"), doc_data.get("processing_status", "PENDING"),
                doc_data.get("security_classification", "INTERNAL"), doc_data.get("source_uri"),
                doc_data.get("mime_type", "text/plain"), doc_data.get("language", "vi"),
                self._to_json(doc_data.get("acl", {})), self._to_json(doc_data.get("metadata", {}))
            ))

            # 2. ContentBlocks
            for b in blocks:
                b_data = self._to_dict(b)
                cursor.execute("""
                    INSERT OR REPLACE INTO platform_content_blocks
                    (block_id, document_id, domain_id, block_index, block_type, text, parent_block_id, source_location, start_offset, end_offset, metadata)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    b_data.get("block_id"), b_data.get("document_id"), b_data.get("domain_id"),
                    b_data.get("block_index", 0), b_data.get("block_type", "PARAGRAPH"),
                    b_data.get("text", ""), b_data.get("parent_block_id"),
                    self._to_json(b_data.get("source_location", {})),
                    b_data.get("start_offset"), b_data.get("end_offset"),
                    self._to_json(b_data.get("metadata", {}))
                ))

            # 3. Chunks
            for c in chunks:
                c_data = self._to_dict(c)
                cursor.execute("""
                    INSERT OR REPLACE INTO platform_chunks
                    (chunk_id, document_id, domain_id, source_block_ids, content, chunk_index, chunking_strategy, profile_id, profile_version, token_count, source_location, security_classification, acl, metadata)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    c_data.get("chunk_id"), c_data.get("document_id"), c_data.get("domain_id"),
                    self._to_json(c_data.get("source_block_ids", [])),
                    c_data.get("content", ""), c_data.get("chunk_index", 0),
                    c_data.get("chunking_strategy", "generic"), c_data.get("profile_id"),
                    c_data.get("profile_version", "1.0.0"), c_data.get("token_count", 0),
                    self._to_json(c_data.get("source_location", {})),
                    c_data.get("security_classification", "INTERNAL"),
                    self._to_json(c_data.get("acl", {})),
                    self._to_json(c_data.get("metadata", {}))
                ))

            # 4. Evidences
            for ev in evidences:
                ev_data = self._to_dict(ev)
                cursor.execute("""
                    INSERT OR REPLACE INTO platform_evidences
                    (evidence_id, domain_id, document_id, block_id, chunk_id, quote_text, source_location, extractor_id, extractor_version, extracted_at, verification_status, origin_type)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    ev_data.get("evidence_id"), ev_data.get("domain_id"), ev_data.get("document_id"),
                    ev_data.get("block_id"), ev_data.get("chunk_id"), ev_data.get("quote_text", ""),
                    self._to_json(ev_data.get("source_location", {})), ev_data.get("extractor_id", "system"),
                    ev_data.get("extractor_version", "1.0.0"), ev_data.get("extracted_at"),
                    ev_data.get("verification_status", "UNVERIFIED"), ev_data.get("origin_type", "EXTRACTOR_GENERATED")
                ))

            # 5. Entities
            for ent in entities:
                ent_data = self._to_dict(ent)
                cursor.execute("""
                    INSERT OR REPLACE INTO platform_entities
                    (entity_id, canonical_name, entity_type, aliases, description, domain_id, graph_namespace, extraction_method, extractor_version, verification_status, schema_version, attributes, evidence_refs)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    ent_data.get("entity_id"), ent_data.get("canonical_name"), ent_data.get("entity_type"),
                    self._to_json(ent_data.get("aliases", [])), ent_data.get("description"),
                    ent_data.get("domain_id"), ent_data.get("graph_namespace"),
                    ent_data.get("extraction_method", "RULE_BASED"), ent_data.get("extractor_version", "1.0.0"),
                    ent_data.get("verification_status", "UNVERIFIED"), ent_data.get("schema_version", "1.0.0"),
                    self._to_json(ent_data.get("attributes", {})),
                    self._to_json(ent_data.get("evidence_refs", []))
                ))

            # 6. Entity Mentions
            for m in mentions:
                m_data = self._to_dict(m)
                cursor.execute("""
                    INSERT OR REPLACE INTO platform_entity_mentions
                    (mention_id, entity_id, domain_id, document_id, block_id, chunk_id, surface_text, source_location, verification_status, evidence_id)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    m_data.get("mention_id"), m_data.get("entity_id"), m_data.get("domain_id"),
                    m_data.get("document_id"), m_data.get("block_id"), m_data.get("chunk_id"),
                    m_data.get("surface_text"), self._to_json(m_data.get("source_location", {})),
                    m_data.get("verification_status", "UNVERIFIED"), m_data.get("evidence_id")
                ))

            # 7. Relationships
            for rel in relationships:
                rel_data = self._to_dict(rel)
                cursor.execute("""
                    INSERT OR REPLACE INTO platform_relationships
                    (relationship_id, subject_entity_id, predicate, object_entity_id, domain_id, graph_namespace, evidence_refs, extraction_method, extractor_version, confidence, verification_status, recorded_at, schema_version, attributes)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    rel_data.get("relationship_id"), rel_data.get("subject_entity_id"),
                    rel_data.get("predicate"), rel_data.get("object_entity_id"),
                    rel_data.get("domain_id"), rel_data.get("graph_namespace"),
                    self._to_json(rel_data.get("evidence_refs", [])),
                    rel_data.get("extraction_method", "RULE_BASED"), rel_data.get("extractor_version", "1.0.0"),
                    float(rel_data.get("confidence", 1.0)), rel_data.get("verification_status", "UNVERIFIED"),
                    rel_data.get("recorded_at"), rel_data.get("schema_version", "1.0.0"),
                    self._to_json(rel_data.get("attributes", {}))
                ))

            conn.commit()

    def get_platform_domains(self, enabled_only: bool = False) -> List[Dict[str, Any]]:
        with self.get_connection() as conn:
            cursor = conn.cursor()
            query = "SELECT * FROM platform_domains"
            params: List[Any] = []
            if enabled_only:
                query += " WHERE enabled = 1"
            cursor.execute(query, params)
            rows = cursor.fetchall()
            results = []
            for r in rows:
                item = dict(r)
                item["allowed_entity_types"] = json.loads(item["allowed_entity_types"]) if item.get("allowed_entity_types") else []
                item["allowed_predicates"] = json.loads(item["allowed_predicates"]) if item.get("allowed_predicates") else []
                item["enabled"] = bool(item.get("enabled", 1))
                item["is_test_fixture"] = bool(item.get("is_test_fixture", 0))
                results.append(item)
            return results

    def get_platform_profiles(self, domain_id: Optional[str] = None, enabled_only: bool = False) -> List[Dict[str, Any]]:
        with self.get_connection() as conn:
            cursor = conn.cursor()
            query = "SELECT * FROM platform_profiles WHERE 1=1"
            params: List[Any] = []
            if domain_id:
                query += " AND (domain_id = ? OR domain_id = '*')"
                params.append(domain_id)
            if enabled_only:
                query += " AND enabled = 1"
            cursor.execute(query, params)
            rows = cursor.fetchall()
            results = []
            for r in rows:
                item = dict(r)
                item["chunking_parameters"] = json.loads(item["chunking_parameters"]) if item.get("chunking_parameters") else {}
                item["metadata_extractors"] = json.loads(item["metadata_extractors"]) if item.get("metadata_extractors") else []
                item["validation_rules"] = json.loads(item["validation_rules"]) if item.get("validation_rules") else {}
                item["enabled"] = bool(item.get("enabled", 1))
                results.append(item)
            return results

    def get_platform_document(self, document_id: str) -> Optional[Dict[str, Any]]:

        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM platform_documents WHERE document_id = ?", (document_id,))
            row = cursor.fetchone()
            if not row:
                return None
            res = dict(row)
            res["acl"] = json.loads(res["acl"]) if res.get("acl") else {}
            res["metadata"] = json.loads(res["metadata"]) if res.get("metadata") else {}
            return res

    def get_platform_blocks(self, document_id: str) -> List[Dict[str, Any]]:
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT * FROM platform_content_blocks WHERE document_id = ? ORDER BY block_index ASC",
                (document_id,)
            )
            rows = cursor.fetchall()
            results = []
            for r in rows:
                item = dict(r)
                item["source_location"] = json.loads(item["source_location"]) if item.get("source_location") else {}
                item["metadata"] = json.loads(item["metadata"]) if item.get("metadata") else {}
                results.append(item)
            return results

    def get_platform_chunks(
        self,
        domain_id: Optional[str] = None,
        profile_id: Optional[str] = None,
        document_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        with self.get_connection() as conn:
            cursor = conn.cursor()
            query = "SELECT * FROM platform_chunks WHERE 1=1"
            params: List[Any] = []
            if domain_id:
                query += " AND domain_id = ?"
                params.append(domain_id)
            if profile_id:
                query += " AND profile_id = ?"
                params.append(profile_id)
            if document_id:
                query += " AND document_id = ?"
                params.append(document_id)
            query += " ORDER BY chunk_index ASC"
            cursor.execute(query, tuple(params))
            results = []
            for r in cursor.fetchall():
                item = dict(r)
                item["source_block_ids"] = json.loads(item["source_block_ids"]) if item.get("source_block_ids") else []
                item["source_location"] = json.loads(item["source_location"]) if item.get("source_location") else {}
                item["acl"] = json.loads(item["acl"]) if item.get("acl") else {}
                item["metadata"] = json.loads(item["metadata"]) if item.get("metadata") else {}
                results.append(item)
            return results

    def get_platform_entities(
        self,
        domain_id: Optional[str] = None,
        graph_namespace: Optional[str] = None,
        verification_status: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        with self.get_connection() as conn:
            cursor = conn.cursor()
            query = "SELECT * FROM platform_entities WHERE 1=1"
            params: List[Any] = []
            if domain_id:
                query += " AND domain_id = ?"
                params.append(domain_id)
            if graph_namespace:
                query += " AND graph_namespace = ?"
                params.append(graph_namespace)
            if verification_status:
                query += " AND verification_status = ?"
                params.append(verification_status)
            cursor.execute(query, tuple(params))
            results = []
            for r in cursor.fetchall():
                item = dict(r)
                item["aliases"] = json.loads(item["aliases"]) if item.get("aliases") else []
                item["attributes"] = json.loads(item["attributes"]) if item.get("attributes") else {}
                item["evidence_refs"] = json.loads(item["evidence_refs"]) if item.get("evidence_refs") else []
                results.append(item)
            return results

    def get_platform_relationships(
        self,
        domain_id: Optional[str] = None,
        graph_namespace: Optional[str] = None,
        include_unverified: bool = False,
    ) -> List[Dict[str, Any]]:
        with self.get_connection() as conn:
            cursor = conn.cursor()
            query = "SELECT * FROM platform_relationships WHERE 1=1"
            params: List[Any] = []
            if domain_id:
                query += " AND domain_id = ?"
                params.append(domain_id)
            if graph_namespace:
                query += " AND graph_namespace = ?"
                params.append(graph_namespace)
            if not include_unverified:
                query += " AND verification_status = 'VERIFIED'"
            cursor.execute(query, tuple(params))
            results = []
            for r in cursor.fetchall():
                item = dict(r)
                item["evidence_refs"] = json.loads(item["evidence_refs"]) if item.get("evidence_refs") else []
                item["attributes"] = json.loads(item["attributes"]) if item.get("attributes") else {}
                results.append(item)
            return results

    def get_platform_evidences(
        self,
        evidence_ids: Optional[List[str]] = None,
        domain_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        with self.get_connection() as conn:
            cursor = conn.cursor()
            query = "SELECT * FROM platform_evidences WHERE 1=1"
            params: List[Any] = []
            if domain_id:
                query += " AND domain_id = ?"
                params.append(domain_id)
            if evidence_ids is not None:
                if not evidence_ids:
                    return []
                placeholders = ",".join(["?"] * len(evidence_ids))
                query += f" AND evidence_id IN ({placeholders})"
                params.extend(evidence_ids)
            cursor.execute(query, tuple(params))
            results = []
            for r in cursor.fetchall():
                item = dict(r)
                item["source_location"] = json.loads(item["source_location"]) if item.get("source_location") else {}
                results.append(item)
            return results

    def get_platform_evidence(self, evidence_id: str) -> Optional[Dict[str, Any]]:
        evs = self.get_platform_evidences(evidence_ids=[evidence_id])
        return evs[0] if evs else None

    def get_platform_entity_mappings(
        self,
        source_namespace: Optional[str] = None,
        target_namespace: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        with self.get_connection() as conn:
            cursor = conn.cursor()
            query = "SELECT * FROM platform_entity_mappings WHERE 1=1"
            params: List[Any] = []
            if source_namespace:
                query += " AND source_namespace = ?"
                params.append(source_namespace)
            if target_namespace:
                query += " AND target_namespace = ?"
                params.append(target_namespace)
            cursor.execute(query, tuple(params))
            results = []
            for r in cursor.fetchall():
                item = dict(r)
                item["metadata"] = json.loads(item["metadata"]) if item.get("metadata") else {}
                results.append(item)
            return results

