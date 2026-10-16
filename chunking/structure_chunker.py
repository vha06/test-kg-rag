import hashlib
from typing import List, Dict
from storage.models import LegalDocumentModel, ChunkModel, ProvisionModel
from chunking.base import BaseChunker


class StructureChunker(BaseChunker):
    def __init__(self, max_tokens: int = 256):
        super().__init__(max_tokens=max_tokens)
        self.strategy_name = "A3_structure"

    def chunk_documents(self, documents: List[LegalDocumentModel]) -> List[ChunkModel]:
        chunks: List[ChunkModel] = []

        for doc in documents:
            articles: List[ProvisionModel] = [p for p in doc.provisions if p.level == "ARTICLE"]
            children_by_parent: Dict[str, List[ProvisionModel]] = {}
            for p in doc.provisions:
                if p.parent_id:
                    children_by_parent.setdefault(p.parent_id, []).append(p)

            for art in articles:
                art_header = f"[Văn bản: {doc.doc_type} số {doc.doc_number} ({doc.title}) | {art.structural_path}]\n{art.title}"
                art_clauses = children_by_parent.get(art.provision_id, [])

                # If short article or no clauses, keep at article level
                if self.count_tokens(art.content) <= self.max_tokens or not art_clauses:
                    full_text = f"{art_header}\n{art.content}".strip()
                    c_id = f"CHK_{self.strategy_name}_{art.provision_id}"
                    chunks.append(ChunkModel(
                        chunk_id=c_id,
                        strategy=self.strategy_name,
                        document_id=doc.document_id,
                        provision_id=art.provision_id,
                        source_provision_ids=[art.provision_id],
                        structural_path=art.structural_path,
                        content=full_text,
                        token_count=self.count_tokens(full_text),
                        crosses_boundary=False,
                        effective_start=art.effective_start,
                        effective_end=art.effective_end,
                        status=art.status,
                        source_url=art.source_url,
                        content_hash=hashlib.sha256(full_text.encode("utf-8")).hexdigest(),
                    ))
                    continue

                # Article is long and has clauses: chunk at Clause or Point level with injected context
                for cl in art_clauses:
                    cl_header = f"{art_header}\n{cl.title}"
                    cl_points = children_by_parent.get(cl.provision_id, [])

                    if self.count_tokens(cl.content) <= self.max_tokens or not cl_points:
                        # Clause chunk
                        cl_full_text = f"{cl_header}\n{cl.content}".strip()
                        c_id = f"CHK_{self.strategy_name}_{cl.provision_id}"
                        chunks.append(ChunkModel(
                            chunk_id=c_id,
                            strategy=self.strategy_name,
                            document_id=doc.document_id,
                            provision_id=cl.provision_id,
                            source_provision_ids=[art.provision_id, cl.provision_id],
                            structural_path=cl.structural_path,
                            content=cl_full_text,
                            token_count=self.count_tokens(cl_full_text),
                            crosses_boundary=False,
                            effective_start=cl.effective_start,
                            effective_end=cl.effective_end,
                            status=cl.status,
                            source_url=cl.source_url,
                            content_hash=hashlib.sha256(cl_full_text.encode("utf-8")).hexdigest(),
                        ))
                    else:
                        # Point chunks
                        for pt in cl_points:
                            pt_full_text = f"{cl_header}\n{pt.title}: {pt.content}".strip()
                            c_id = f"CHK_{self.strategy_name}_{pt.provision_id}"
                            chunks.append(ChunkModel(
                                chunk_id=c_id,
                                strategy=self.strategy_name,
                                document_id=doc.document_id,
                                provision_id=pt.provision_id,
                                source_provision_ids=[art.provision_id, cl.provision_id, pt.provision_id],
                                structural_path=pt.structural_path,
                                content=pt_full_text,
                                token_count=self.count_tokens(pt_full_text),
                                crosses_boundary=False,
                                effective_start=pt.effective_start,
                                effective_end=pt.effective_end,
                                status=pt.status,
                                source_url=pt.source_url,
                                content_hash=hashlib.sha256(pt_full_text.encode("utf-8")).hexdigest(),
                            ))

        return chunks
