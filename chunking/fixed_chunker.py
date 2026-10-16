import hashlib
import re
from typing import List, Dict, Any, Tuple
from storage.models import LegalDocumentModel, ChunkModel
from chunking.base import BaseChunker


class FixedChunker(BaseChunker):
    def __init__(self, chunk_size: int = 256, chunk_overlap: int = 32):
        super().__init__(max_tokens=chunk_size)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.strategy_name = "A2_fixed"

    def chunk_documents(self, documents: List[LegalDocumentModel]) -> List[ChunkModel]:
        chunks: List[ChunkModel] = []

        for doc in documents:
            articles = [p for p in doc.provisions if p.level == "ARTICLE"]
            if not articles:
                continue

            # Build full token stream with token-to-provision tracking
            token_stream: List[Tuple[str, str, str]] = []  # (token, provision_id, path)
            for art in articles:
                art_header = f"[{doc.doc_type} số {doc.doc_number}] {art.title}: "
                art_tokens = re.findall(r"\w+|[^\w\s]", art_header + art.content, re.UNICODE)
                for t in art_tokens:
                    token_stream.append((t, art.provision_id, art.structural_path))

            total_tokens = len(token_stream)
            step = max(1, self.chunk_size - self.chunk_overlap)
            idx = 0
            chunk_num = 1

            while idx < total_tokens:
                window = token_stream[idx : idx + self.chunk_size]
                if not window:
                    break

                text_tokens = [w[0] for w in window]
                # Reconstruct text spacing
                chunk_text = " ".join(text_tokens)

                # Collect touched provisions
                prov_counts: Dict[str, int] = {}
                prov_paths: Dict[str, str] = {}
                for _, prov_id, p_path in window:
                    prov_counts[prov_id] = prov_counts.get(prov_id, 0) + 1
                    prov_paths[prov_id] = p_path

                # Primary provision is the one with majority tokens
                primary_prov_id = max(prov_counts.items(), key=lambda x: x[1])[0]
                primary_path = prov_paths[primary_prov_id]

                # Boundary crosses if > 1 provision touched
                crosses_boundary = len(prov_counts) > 1

                c_id = f"CHK_{self.strategy_name}_{doc.document_id}_W{chunk_num}"
                c_hash = hashlib.sha256(chunk_text.encode("utf-8")).hexdigest()

                chunks.append(ChunkModel(
                    chunk_id=c_id,
                    strategy=self.strategy_name,
                    document_id=doc.document_id,
                    provision_id=primary_prov_id,
                    source_provision_ids=list(prov_counts.keys()),
                    structural_path=primary_path,
                    content=chunk_text,
                    token_count=len(window),
                    crosses_boundary=crosses_boundary,
                    effective_start=doc.effective_start,
                    effective_end=doc.effective_end,
                    status=doc.status,
                    source_url=doc.source_url,
                    content_hash=c_hash,
                ))

                chunk_num += 1
                idx += step

        return chunks
