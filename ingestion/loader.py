import os
import sys
import json
from typing import List, Tuple, Optional

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass
from storage.models import LegalDocumentModel, KGEdgeModel
from storage.db import DatabaseManager
from ingestion.phapdien_converter import PhapdienConverter


class IngestionLoader:
    def __init__(self, db_manager: Optional[DatabaseManager] = None):
        self.db = db_manager or DatabaseManager()

    def ingest_from_jsonl(
        self,
        docs_path: str = "data/processed/vbqppl_documents.jsonl",
        edges_path: str = "data/processed/kg_relations.jsonl",
    ) -> Tuple[int, int]:
        if not os.path.exists(docs_path) or not os.path.exists(edges_path):
            print(f"Processed files not found. Running PhapdienConverter first...")
            converter = PhapdienConverter()
            converter.convert(
                raw_json_path="mau_data.json",
                output_docs_path=docs_path,
                output_edges_path=edges_path,
            )

        print(f"Loading documents from {docs_path}...")
        docs: List[LegalDocumentModel] = []
        with open(docs_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    docs.append(LegalDocumentModel.model_validate_json(line.strip()))

        print(f"Loading edges from {edges_path}...")
        edges: List[KGEdgeModel] = []
        with open(edges_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    edges.append(KGEdgeModel.model_validate_json(line.strip()))

        print(f"Ingesting {len(docs)} documents and {len(edges)} edges into SQLite...")
        self.db.insert_documents(docs)
        self.db.insert_kg_edges(edges)

        summary = self.db.count_summary()
        print(f"Database status after ingestion: {summary}")
        return len(docs), len(edges)


if __name__ == "__main__":
    loader = IngestionLoader()
    loader.ingest_from_jsonl()
