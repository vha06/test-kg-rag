import os
import sys
import time

# Ensure repository root is on sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from storage.db import DatabaseManager
from storage.models import LegalDocumentModel
from ingestion.loader import IngestionLoader
from chunking.article_chunker import ArticleChunker
from chunking.fixed_chunker import FixedChunker
from chunking.structure_chunker import StructureChunker
from chunking.base import BaseChunker
from retrieval.bm25_retriever import BM25Retriever
from embedding.adapter import EmbeddingAdapter
from embedding.index_manager import FAISSIndexManager
from knowledge_graph.builder import KnowledgeGraphBuilder
from knowledge_graph.citation_extractor import CitationExtractor
from evaluation.benchmark_builder import BenchmarkBuilder


def main():
    print("=== STEP 1: INGESTION & DATA CONVERSION ===")
    db = DatabaseManager()
    loader = IngestionLoader(db)
    loader.ingest_from_jsonl()

    print("\n=== STEP 2: KNOWLEDGE GRAPH ENRICHMENT ===")
    all_doc_numbers = {}
    with db.get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT document_id, doc_number FROM documents")
        for row in cursor.fetchall():
            all_doc_numbers[row["doc_number"]] = row["document_id"]

        # Fetch articles to extract DETAILS and in-text references
        cursor.execute("""
            SELECT p.*, d.doc_number
            FROM provisions p
            JOIN documents d ON p.document_id = d.document_id
            WHERE p.level = 'ARTICLE'
        """)
        articles = [dict(r) for r in cursor.fetchall()]

    print(f"Extracting additional citations across {len(articles)} articles...")
    new_edges = []
    doc_first_articles = {}
    for a in articles:
        doc_id = a["document_id"]
        if doc_id not in doc_first_articles:
            doc_first_articles[doc_id] = a

    for doc_id, fa in doc_first_articles.items():
        det_edges = CitationExtractor.extract_details_relations(
            doc_id=doc_id,
            doc_number=fa["doc_number"],
            first_article_text=fa["content"],
            ghichu_text=fa["title"],
            all_doc_numbers=all_doc_numbers,
        )
        new_edges.extend(det_edges)

    print(f"Extracted {len(new_edges)} additional DETAILS relations.")
    db.insert_kg_edges(new_edges)

    print("\n=== STEP 3: CHUNKING & INDEXING (A1, A2, A3) ===")
    # Load all documents from SQLite
    docs_to_chunk: list[LegalDocumentModel] = []
    with open("data/processed/vbqppl_documents.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                docs_to_chunk.append(LegalDocumentModel.model_validate_json(line.strip()))

    chunkers = [
        ArticleChunker(max_tokens=512),
        FixedChunker(chunk_size=256, chunk_overlap=32),
        StructureChunker(max_tokens=256),
    ]

    adapter = EmbeddingAdapter()

    for chunker in chunkers:
        strat = chunker.strategy_name
        print(f"\n--- Processing Strategy: {strat} ---")
        t0 = time.time()
        chunks = chunker.chunk_documents(docs_to_chunk)
        stats = BaseChunker.compute_stats(chunks)
        print(f"Generated {len(chunks)} chunks in {time.time() - t0:.2f}s.")
        print(f"Stats: {stats}")

        print(f"Inserting chunks into SQLite...")
        db.insert_chunks(chunks)

        # Build BM25 Index
        print(f"Building BM25 index for {strat}...")
        bm25 = BM25Retriever(strategy=strat)
        db_chunks = db.get_chunks_by_strategy(strat)
        bm25.build_index(db_chunks)

        # Build FAISS Index
        print(f"Building FAISS vector index for {strat}...")
        faiss_mgr = FAISSIndexManager(strategy=strat, adapter=adapter)
        faiss_mgr.build_index(db_chunks)

    print("\n=== STEP 4: BENCHMARK DATASET GENERATION ===")
    bench_builder = BenchmarkBuilder(db)
    bench_builder.build_default_benchmark()

    print("\n=== ALL DATA, INDICES & BENCHMARK ARE READY! ===")
    print("Database counts:", db.count_summary())


if __name__ == "__main__":
    main()
