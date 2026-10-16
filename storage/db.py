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
            for table in ["documents", "provisions", "provision_versions", "chunks", "kg_edges"]:
                cursor.execute(f"SELECT COUNT(*) FROM {table}")
                counts[table] = cursor.fetchone()[0]
            return counts
