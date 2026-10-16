import hashlib
from typing import List
from storage.models import LegalDocumentModel, ChunkModel, ProvisionModel
from chunking.base import BaseChunker


class ArticleChunker(BaseChunker):
    def __init__(self, max_tokens: int = 512):
        super().__init__(max_tokens=max_tokens)
        self.strategy_name = "A1_article"

    def chunk_documents(self, documents: List[LegalDocumentModel]) -> List[ChunkModel]:
        chunks: List[ChunkModel] = []

        for doc in documents:
            # Map articles and their child clauses/points
            articles: List[ProvisionModel] = [p for p in doc.provisions if p.level == "ARTICLE"]
            children_by_parent: dict = {}
            for p in doc.provisions:
                if p.parent_id:
                    children_by_parent.setdefault(p.parent_id, []).append(p)

            for art in articles:
                header = f"[{doc.doc_type} số {doc.doc_number} ({doc.title}) | {art.structural_path}] {art.title}"
                full_text = f"{header}\n{art.content}".strip()
                t_count = self.count_tokens(full_text)

                if t_count <= self.max_tokens or not children_by_parent.get(art.provision_id):
                    # Single chunk for this article
                    c_id = f"CHK_{self.strategy_name}_{art.provision_id}"
                    c_hash = hashlib.sha256(full_text.encode("utf-8")).hexdigest()
                    chunks.append(ChunkModel(
                        chunk_id=c_id,
                        strategy=self.strategy_name,
                        document_id=doc.document_id,
                        provision_id=art.provision_id,
                        source_provision_ids=[art.provision_id],
                        structural_path=art.structural_path,
                        content=full_text,
                        token_count=t_count,
                        crosses_boundary=False,
                        effective_start=art.effective_start,
                        effective_end=art.effective_end,
                        status=art.status,
                        source_url=art.source_url,
                        content_hash=c_hash,
                    ))
                else:
                    # Article is too long: split along clause boundaries without losing text
                    clauses = children_by_parent.get(art.provision_id, [])
                    cur_clause_texts = []
                    cur_prov_ids = [art.provision_id]
                    part_idx = 1

                    for cl in clauses:
                        cl_text = cl.content
                        combined = f"{header}\n" + "\n".join(cur_clause_texts + [cl_text])
                        if self.count_tokens(combined) > self.max_tokens and cur_clause_texts:
                            # Flush current part
                            part_text = f"{header} (Phần {part_idx})\n" + "\n".join(cur_clause_texts)
                            c_id = f"CHK_{self.strategy_name}_{art.provision_id}_P{part_idx}"
                            chunks.append(ChunkModel(
                                chunk_id=c_id,
                                strategy=self.strategy_name,
                                document_id=doc.document_id,
                                provision_id=art.provision_id,
                                source_provision_ids=list(cur_prov_ids),
                                structural_path=f"{art.structural_path} (Phần {part_idx})",
                                content=part_text,
                                token_count=self.count_tokens(part_text),
                                crosses_boundary=False,
                                effective_start=art.effective_start,
                                effective_end=art.effective_end,
                                status=art.status,
                                source_url=art.source_url,
                                content_hash=hashlib.sha256(part_text.encode("utf-8")).hexdigest(),
                            ))
                            part_idx += 1
                            cur_clause_texts = [cl_text]
                            cur_prov_ids = [art.provision_id, cl.provision_id]
                        else:
                            cur_clause_texts.append(cl_text)
                            cur_prov_ids.append(cl.provision_id)

                    if cur_clause_texts:
                        part_text = f"{header} (Phần {part_idx})\n" + "\n".join(cur_clause_texts)
                        c_id = f"CHK_{self.strategy_name}_{art.provision_id}_P{part_idx}"
                        chunks.append(ChunkModel(
                            chunk_id=c_id,
                            strategy=self.strategy_name,
                            document_id=doc.document_id,
                            provision_id=art.provision_id,
                            source_provision_ids=list(cur_prov_ids),
                            structural_path=f"{art.structural_path} (Phần {part_idx})",
                            content=part_text,
                            token_count=self.count_tokens(part_text),
                            crosses_boundary=False,
                            effective_start=art.effective_start,
                            effective_end=art.effective_end,
                            status=art.status,
                            source_url=art.source_url,
                            content_hash=hashlib.sha256(part_text.encode("utf-8")).hexdigest(),
                        ))

        return chunks
