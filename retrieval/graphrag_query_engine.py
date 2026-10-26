from __future__ import annotations

import time
import uuid
import logging
from typing import List, Dict, Any, Optional, Set, Tuple

from storage.db import DatabaseManager
from retrieval.graphrag_index_manager import GraphRAGIndexManager
from core.schema import (
    GraphRAGQueryResult,
    GraphRAGQueryTrace,
    CanonicalChunk,
    CanonicalEntity,
    CanonicalRelationship,
    CanonicalCommunityReport,
)

logger = logging.getLogger(__name__)


class GraphRAGQueryEngine:
    """
    Bộ máy thực thi 4 chế độ truy vấn tiêu chuẩn theo kiến trúc Microsoft GraphRAG:
    1. Vector RAG Baseline: Truy xuất thuần vector trên TextUnits (Chunks)
    2. Local Search: Entity Linking -> Subgraph Expansion -> Linked TextUnits -> Tổng hợp câu trả lời
    3. Global Search: Community Report Selection -> Map (lọc luận điểm) -> Reduce (tổng hợp toàn cục kèm TextUnits gốc)
    4. DRIFT Search: Community Primer -> Sinh câu hỏi phụ -> Duyệt đệ quy Local Search có dừng sớm (Information-gain Early Stopping)
    """

    REFUSAL_MESSAGE = "Không đủ căn cứ trong tài liệu được cung cấp để trả lời câu hỏi này."

    def __init__(
        self,
        db_manager: DatabaseManager,
        index_manager: GraphRAGIndexManager,
        llm_config: Optional[Any] = None,
    ):
        self.db = db_manager
        self.index_manager = index_manager
        self.llm_config = llm_config
        self.last_thought: Optional[str] = None

    def _call_llm(self, system_prompt: str, user_prompt: str) -> str:
        self.last_thought = None
        is_mock = (
            self.llm_config is None
            or getattr(self.llm_config, "provider", "mock") == "mock"
            or not getattr(self.llm_config, "api_key", None) and getattr(self.llm_config, "provider", "") == "gemini"
        )
        if is_mock:
            return ""

        from generation.llm_client import MultiProviderLLMClient
        client = MultiProviderLLMClient(self.llm_config)
        ans, thought = client.generate_with_thought(system_prompt=system_prompt, user_prompt=user_prompt)
        if thought:
            self.last_thought = thought
        return ans

    # =========================================================================
    # CHẾ ĐỘ 1: VECTOR RAG BASELINE
    # =========================================================================

    def query_vector_rag(
        self,
        query: str,
        namespace: str = "graphrag_default",
        top_k: int = 5,
        dense_weight: float = 0.5,
    ) -> GraphRAGQueryResult:
        start_time = time.perf_counter()
        query_id = f"QRY_VEC_{uuid.uuid4().hex[:8]}"

        results = self.index_manager.search_text_units(
            query=query,
            namespace=namespace,
            top_k=top_k,
            dense_weight=dense_weight,
        )

        selected_chunk_ids = [r["chunk_id"] for r in results]
        citations = [
            {
                "citation_id": r["chunk_id"],
                "source_type": "TEXT_UNIT",
                "document_id": r.get("document_id", ""),
                "score": r.get("score", 0.0),
                "content_snippet": r.get("content", "")[:150],
            }
            for r in results
        ]

        if not results:
            trace = GraphRAGQueryTrace(
                query_id=query_id,
                query_text=query,
                query_mode="VECTOR",
                stop_reason="NO_RESULTS",
                refusal_reason="NO_CHUNKS_FOUND",
                latency_ms=(time.perf_counter() - start_time) * 1000,
            )
            return GraphRAGQueryResult(
                query_id=query_id,
                query_text=query,
                query_mode="VECTOR",
                answer=self.REFUSAL_MESSAGE,
                grounded_citations=[],
                trace=trace,
            )

        context_blocks = "\n\n".join([f"[{r['chunk_id']}]: {r['content']}" for r in results])
        sys_prompt = (
            "Bạn là trợ lý giải đáp câu hỏi dựa trên các đoạn văn bản (TextUnits) được cung cấp. "
            "Chỉ trả lời dựa trên ngữ cảnh, bắt buộc trích dẫn mã [Chunk_ID] ở các thông tin lấy từ tài liệu. "
            f"Nếu ngữ cảnh không có thông tin để trả lời, hãy trả lời chính xác: '{self.REFUSAL_MESSAGE}'."
        )
        user_prompt = f"Ngữ cảnh:\n{context_blocks}\n\nCâu hỏi: {query}\n\nTrả lời:"

        llm_resp = self._call_llm(sys_prompt, user_prompt)
        llm_calls = 1 if llm_resp else 0

        if not llm_resp:
            # Deterministic answer synthesis in mock mode
            primary = results[0]
            answer = f"Theo thông tin từ tài liệu [{primary['chunk_id']}]: {primary['content'][:300]}"

        else:
            answer = llm_resp.strip()

        trace = GraphRAGQueryTrace(
            query_id=query_id,
            query_text=query,
            query_mode="VECTOR",
            selected_chunk_ids=selected_chunk_ids,
            llm_calls=llm_calls,
            latency_ms=round((time.perf_counter() - start_time) * 1000, 2),
            stop_reason="SUCCESS",
            details={"thinking": self.last_thought} if self.last_thought else {},
        )

        return GraphRAGQueryResult(
            query_id=query_id,
            query_text=query,
            query_mode="VECTOR",
            answer=answer,
            grounded_citations=citations,
            trace=trace,
        )

    # =========================================================================
    # CHẾ ĐỘ 2: LOCAL SEARCH
    # =========================================================================

    def query_local_search(
        self,
        query: str,
        namespace: str = "graphrag_default",
        top_k_entities: int = 5,
        max_hops: int = 1,
        top_k_chunks: int = 5,
        max_context_chunks: Optional[int] = None,
        dense_weight: float = 0.5,
    ) -> GraphRAGQueryResult:
        if max_context_chunks is not None:
            top_k_chunks = max_context_chunks
        start_time = time.perf_counter()
        query_id = f"QRY_LOC_{uuid.uuid4().hex[:8]}"

        # 1. Entity Linking
        seed_entities = self.index_manager.search_entities(
            query=query,
            namespace=namespace,
            top_k=top_k_entities,
            dense_weight=dense_weight,
        )

        if not seed_entities:
            trace = GraphRAGQueryTrace(
                query_id=query_id,
                query_text=query,
                query_mode="LOCAL",
                stop_reason="NO_SEEDS",
                refusal_reason="NO_ENTITIES_LINKED",
                latency_ms=(time.perf_counter() - start_time) * 1000,
            )
            return GraphRAGQueryResult(
                query_id=query_id,
                query_text=query,
                query_mode="LOCAL",
                answer=self.REFUSAL_MESSAGE,
                grounded_citations=[],
                trace=trace,
            )

        seed_ids = [e["entity_id"] for e in seed_entities]

        # 2. Subgraph Expansion
        all_relationships = self.db.get_platform_relationships(graph_namespace=namespace, include_unverified=True)
        all_entities = self.db.get_platform_entities(graph_namespace=namespace)
        ent_by_id = {e["entity_id"]: e for e in all_entities}

        expanded_node_ids: Set[str] = set(seed_ids)
        selected_rel_ids: List[str] = []

        frontier = set(seed_ids)
        for _ in range(max_hops):
            next_frontier = set()
            for r in all_relationships:
                src = r["subject_entity_id"]
                tgt = r["object_entity_id"]
                if src in frontier and tgt not in expanded_node_ids:
                    expanded_node_ids.add(tgt)
                    next_frontier.add(tgt)
                    selected_rel_ids.append(r["relationship_id"])
                elif tgt in frontier and src not in expanded_node_ids:
                    expanded_node_ids.add(src)
                    next_frontier.add(src)
                    selected_rel_ids.append(r["relationship_id"])
                elif src in frontier and tgt in frontier:
                    selected_rel_ids.append(r["relationship_id"])
            frontier = next_frontier

        selected_rel_ids = list(dict.fromkeys(selected_rel_ids))

        # 3. Linked TextUnits (Chunks)
        linked_chunk_ids: List[str] = []
        for nid in expanded_node_ids:
            if nid in ent_by_id:
                linked_chunk_ids.extend(ent_by_id[nid].get("source_chunk_ids", []))
        for rid in selected_rel_ids:
            rel_obj = next((r for r in all_relationships if r["relationship_id"] == rid), None)
            if rel_obj:
                linked_chunk_ids.extend(rel_obj.get("source_chunk_ids", []))

        linked_chunk_ids = list(dict.fromkeys(linked_chunk_ids))

        # Retrieve chunk records
        all_chunks = self.db.get_platform_chunks(domain_id=namespace)
        if not all_chunks:
            all_chunks = self.db.get_platform_chunks()
        chunk_map = {c["chunk_id"]: c for c in all_chunks}
        matched_chunks = [chunk_map[cid] for cid in linked_chunk_ids if cid in chunk_map]

        if not matched_chunks:
            # Fallback to vector search on text units
            matched_chunks = self.index_manager.search_text_units(query, namespace, top_k=top_k_chunks)

        selected_chunks = matched_chunks[:top_k_chunks]
        selected_chunk_ids = [c["chunk_id"] for c in selected_chunks]

        # 4. Context Assembly
        ent_lines = [
            f"- {ent_by_id[nid].get('canonical_name', nid)} ({ent_by_id[nid].get('entity_type', 'ENTITY')}): {ent_by_id[nid].get('description', '')}"
            for nid in expanded_node_ids if nid in ent_by_id
        ]
        rel_lines = [
            f"- {r.get('description') or (r['subject_entity_id'] + ' ' + r['predicate'] + ' ' + r['object_entity_id'])}"
            for r in all_relationships if r["relationship_id"] in selected_rel_ids
        ]
        chunk_lines = [f"[{c['chunk_id']}]: {c.get('content', '')}" for c in selected_chunks]

        context_str = (
            "THỰC THỂ LIÊN QUAN:\n" + "\n".join(ent_lines) + "\n\n"
            "MỐI QUAN HỆ TRONG ĐỒ THỊ:\n" + "\n".join(rel_lines) + "\n\n"
            "ĐOẠN VĂN BẢN GỐC (TEXT UNITS):\n" + "\n".join(chunk_lines)
        )

        sys_prompt = (
            "Bạn là trợ lý Local Search chuyên sâu trên Đồ thị tri thức và tài liệu gốc. "
            "Hãy tổng hợp câu trả lời dựa trên các thực thể, mối quan hệ và đoạn văn bản gốc được cung cấp. "
            "Bắt buộc trích dẫn các mã [Chunk_ID] của đoạn văn bản làm cơ sở cho từng luận điểm. "
            f"Nếu thông tin không đủ để kết luận, hãy trả lời chính xác: '{self.REFUSAL_MESSAGE}'."
        )
        user_prompt = f"Ngữ cảnh trích xuất:\n{context_str}\n\nCâu hỏi: {query}\n\nTrả lời:"

        llm_resp = self._call_llm(sys_prompt, user_prompt)
        llm_calls = 1 if llm_resp else 0

        if not llm_resp:
            # Deterministic synthesis
            lead_ent = seed_entities[0]["canonical_name"]
            top_rel = rel_lines[0] if rel_lines else "có mối quan hệ liên kết trong đồ thị"
            top_chunk_id = selected_chunk_ids[0] if selected_chunk_ids else "CHUNK"
            chunk_content = selected_chunks[0].get("content", "")[:250] if selected_chunks else ""
            answer = (
                f"Dựa trên đồ thị tri thức xoay quanh thực thể '{lead_ent}', hệ thống ghi nhận: {top_rel}. "
                f"Căn cứ đoạn trích [{top_chunk_id}]: {chunk_content}"
            )
        else:
            answer = llm_resp.strip()

        citations = [
            {
                "citation_id": c["chunk_id"],
                "source_type": "TEXT_UNIT",
                "document_id": c.get("document_id", ""),
                "content_snippet": c.get("content", "")[:150],
            }
            for c in selected_chunks
        ]

        trace = GraphRAGQueryTrace(
            query_id=query_id,
            query_text=query,
            query_mode="LOCAL",
            seed_entity_ids=seed_ids,
            expanded_entity_ids=list(expanded_node_ids),
            selected_relationship_ids=selected_rel_ids,
            selected_chunk_ids=selected_chunk_ids,
            llm_calls=llm_calls,
            latency_ms=round((time.perf_counter() - start_time) * 1000, 2),
            stop_reason="SUCCESS",
            details={"thinking": self.last_thought} if self.last_thought else {},
        )

        return GraphRAGQueryResult(
            query_id=query_id,
            query_text=query,
            query_mode="LOCAL",
            answer=answer,
            grounded_citations=citations,
            trace=trace,
        )

    # =========================================================================
    # CHẾ ĐỘ 3: GLOBAL SEARCH (MAP-REDUCE)
    # =========================================================================

    def query_global_search(
        self,
        query: str,
        namespace: str = "graphrag_default",
        community_level: Optional[int] = 0,
        top_k_communities: int = 5,
        top_k_reports: Optional[int] = None,
        dense_weight: float = 0.5,
    ) -> GraphRAGQueryResult:
        if top_k_reports is not None:
            top_k_communities = top_k_reports
        start_time = time.perf_counter()
        query_id = f"QRY_GLO_{uuid.uuid4().hex[:8]}"

        # 1. Community Report Selection
        reports = self.index_manager.search_community_reports(
            query=query,
            namespace=namespace,
            level=community_level,
            top_k=top_k_communities,
            dense_weight=dense_weight,
        )

        if not reports and community_level is not None:
            # Fallback to any level if specific level has no reports
            reports = self.index_manager.search_community_reports(
                query=query,
                namespace=namespace,
                level=None,
                top_k=top_k_communities,
                dense_weight=dense_weight,
            )

        if not reports:
            trace = GraphRAGQueryTrace(
                query_id=query_id,
                query_text=query,
                query_mode="GLOBAL",
                stop_reason="NO_COMMUNITIES",
                refusal_reason="NO_COMMUNITY_REPORTS_FOUND",
                latency_ms=(time.perf_counter() - start_time) * 1000,
            )
            return GraphRAGQueryResult(
                query_id=query_id,
                query_text=query,
                query_mode="GLOBAL",
                answer=self.REFUSAL_MESSAGE,
                grounded_citations=[],
                trace=trace,
            )

        selected_comm_ids = [r["community_id"] for r in reports]
        accumulated_chunk_ids: List[str] = []
        for r in reports:
            accumulated_chunk_ids.extend(r.get("source_chunk_ids", []))
        accumulated_chunk_ids = list(dict.fromkeys(accumulated_chunk_ids))

        # 2. Map Step
        map_outputs: List[Dict[str, Any]] = []
        llm_calls = 0

        for r in reports:
            comm_title = r.get("title", "")
            comm_summary = r.get("summary", "")
            comm_content = r.get("full_content", "")
            findings = r.get("key_findings", [])

            map_prompt = (
                f"Đánh giá mức độ liên quan của báo cáo cộng đồng '{comm_title}' đối với câu hỏi sau:\n"
                f"Câu hỏi: {query}\n\n"
                f"Nội dung báo cáo:\n{comm_summary}\n\n"
                "Hãy trích xuất các luận điểm chính liên quan đến câu hỏi và cho điểm mức độ liên quan từ 0 đến 100."
            )
            # Use deterministic heuristic if mock, or LLM
            llm_map = self._call_llm("Bạn là chuyên gia thẩm định luận điểm báo cáo.", map_prompt)
            if llm_map:
                llm_calls += 1
                point_text = llm_map.strip()
                score = 80.0
            else:
                point_text = f"Cộng đồng [{r['community_id']} - {comm_title}] tóm lược: {comm_summary}. Các phát hiện: {'; '.join(findings[:2])}"
                score = float(r.get("score", 0.8)) * 100.0

            map_outputs.append({
                "community_id": r["community_id"],
                "report_id": r.get("report_id", ""),
                "title": comm_title,
                "point": point_text,
                "score": score,
                "chunk_ids": r.get("source_chunk_ids", [])[:3],
            })

        # Sắp xếp luận điểm Map theo score
        map_outputs.sort(key=lambda x: x["score"], reverse=True)

        # 3. Reduce Step + Provenance Back-mapping
        all_chunks = self.db.get_platform_chunks()
        chunk_lookup = {c["chunk_id"]: c for c in all_chunks}
        sample_chunks = [chunk_lookup[cid] for cid in accumulated_chunk_ids[:5] if cid in chunk_lookup]
        sample_chunk_citations = [f"[{c['chunk_id']}]: {c.get('content', '')[:200]}" for c in sample_chunks]

        reduce_context = (
            "CÁC LUẬN ĐIỂM TỔNG HỢP TỪ BÁO CÁO CỘNG ĐỒNG (MAP STEP):\n"
            + "\n".join([f"- [Cộng đồng {m['community_id']}]: {m['point']}" for m in map_outputs[:5]])
            + "\n\nĐOẠN TRÍCH NGUYÊN VĂN GỐC (EVIDENCE TEXT UNITS):\n"
            + "\n".join(sample_chunk_citations)
        )

        reduce_sys = (
            "Bạn là tổng hợp viên toàn cục (Global Synthesis Engine) của hệ thống GraphRAG. "
            "Nhiệm vụ: Tổng hợp các luận điểm từ các cộng đồng thành một câu trả lời hoàn chỉnh, mang tính hệ thống. "
            "Bắt buộc gắn kèm trích dẫn mã [Chunk_ID] của tài liệu gốc tương ứng. "
            f"Nếu các cộng đồng không chứa thông tin phù hợp, hãy trả lời chính xác: '{self.REFUSAL_MESSAGE}'."
        )
        reduce_user = f"Ngữ cảnh:\n{reduce_context}\n\nCâu hỏi: {query}\n\nTổng hợp toàn cục:"

        llm_reduce = self._call_llm(reduce_sys, reduce_user)
        if llm_reduce:
            llm_calls += 1
            answer = llm_reduce.strip()
        else:
            top_point = map_outputs[0]["point"]
            ref_chunk = sample_chunks[0]["chunk_id"] if sample_chunks else "CHUNK_REF"
            answer = (
                f"Tổng hợp toàn cục từ các cộng đồng tri thức: {top_point}. "
                f"Căn cứ đối chiếu từ các phân đoạn tài liệu gốc [{ref_chunk}]."
            )

        citations = [
            {
                "citation_id": c["chunk_id"],
                "source_type": "TEXT_UNIT",
                "document_id": c.get("document_id", ""),
                "content_snippet": c.get("content", "")[:150],
            }
            for c in sample_chunks
        ]

        trace = GraphRAGQueryTrace(
            query_id=query_id,
            query_text=query,
            query_mode="GLOBAL",
            selected_community_ids=selected_comm_ids,
            selected_chunk_ids=[c["chunk_id"] for c in sample_chunks],
            map_outputs=map_outputs,
            llm_calls=llm_calls,
            latency_ms=round((time.perf_counter() - start_time) * 1000, 2),
            stop_reason="SUCCESS",
            details={"thinking": self.last_thought} if self.last_thought else {},
        )

        return GraphRAGQueryResult(
            query_id=query_id,
            query_text=query,
            query_mode="GLOBAL",
            answer=answer,
            grounded_citations=citations,
            trace=trace,
        )

    # =========================================================================
    # CHẾ ĐỘ 4: DRIFT SEARCH (HYBRID ITERATIVE EXPANSION)
    # =========================================================================

    def query_drift_search(
        self,
        query: str,
        namespace: str = "graphrag_default",
        max_depth: int = 2,
        max_followups_per_step: int = 2,
        max_total_llm_calls: int = 6,
    ) -> GraphRAGQueryResult:
        start_time = time.perf_counter()
        query_id = f"QRY_DRI_{uuid.uuid4().hex[:8]}"

        total_llm_calls = 0
        drift_steps: List[Dict[str, Any]] = []

        # 1. Primer Step: Global Community Perspective
        primer_reports = self.index_manager.search_community_reports(
            query=query,
            namespace=namespace,
            top_k=2,
            dense_weight=0.5,
        )

        primer_text = ""
        followup_questions: List[str] = []

        if primer_reports:
            p_rep = primer_reports[0]
            primer_text = f"Tổng quan từ cụm '{p_rep.get('title')}': {p_rep.get('summary')}"
            # Sinh 1-2 câu hỏi làm rõ
            followup_prompt = (
                f"Dựa trên câu hỏi gốc '{query}' và thông tin tổng quan sau:\n{primer_text}\n"
                f"Hãy đặt đúng {max_followups_per_step} câu hỏi phụ ngắn gọn để điều tra các chi tiết cụ thể còn thiếu. "
                "Mỗi câu hỏi trên một dòng."
            )
            llm_follow = self._call_llm("Bạn là chuyên gia điều tra DRIFT Search.", followup_prompt)
            if llm_follow:
                total_llm_calls += 1
                lines = [line.strip().lstrip("-*1234. ") for line in llm_follow.splitlines() if line.strip()]
                followup_questions = lines[:max_followups_per_step]
            else:
                followup_questions = [
                    f"Chi tiết quy định hoặc thành phần liên quan đến {query} là gì?",
                    f"Các điều kiện hoặc nguyên nhân thực hiện đối với {query} quy định thế nào?",
                ][:max_followups_per_step]
        else:
            primer_text = "Không tìm thấy báo cáo cộng đồng vĩ mô; khởi tạo trực tiếp truy vấn chi tiết."
            followup_questions = [query]

        drift_steps.append({
            "step": 0,
            "type": "PRIMER",
            "primer_text": primer_text,
            "followups": followup_questions,
        })

        # 2. Iterative Local Search Exploration
        accumulated_chunks: Dict[str, Dict[str, Any]] = {}
        accumulated_entities: Set[str] = set()
        stop_reason = "SUCCESS"

        current_questions = list(followup_questions)
        depth = 0

        while depth < max_depth and current_questions:
            depth += 1
            next_step_questions: List[str] = []
            new_chunks_in_step = 0
            new_entities_in_step = 0

            for sub_q in current_questions:
                if total_llm_calls >= max_total_llm_calls:
                    stop_reason = "MAX_LLM_CALLS_REACHED"
                    break

                # Execute Local Search context retrieval for the sub-question
                loc_res = self.query_local_search(
                    query=sub_q,
                    namespace=namespace,
                    top_k_entities=3,
                    top_k_chunks=3,
                )
                total_llm_calls += loc_res.trace.llm_calls

                # Track information gain
                for cid in loc_res.trace.selected_chunk_ids:
                    if cid not in accumulated_chunks:
                        new_chunks_in_step += 1
                        # Save snippet
                        matching_c = next((c for c in loc_res.grounded_citations if c["citation_id"] == cid), None)
                        accumulated_chunks[cid] = matching_c or {"citation_id": cid, "content_snippet": ""}

                for eid in loc_res.trace.seed_entity_ids + loc_res.trace.expanded_entity_ids:
                    if eid not in accumulated_entities:
                        new_entities_in_step += 1
                        accumulated_entities.add(eid)

                drift_steps.append({
                    "step": depth,
                    "sub_query": sub_q,
                    "new_chunks_count": new_chunks_in_step,
                    "new_entities_count": new_entities_in_step,
                    "answer_snippet": loc_res.answer[:120],
                })

            # Information-Gain Early Stopping
            if new_chunks_in_step == 0 and new_entities_in_step == 0:
                stop_reason = "STOPPED_NO_NEW_EVIDENCE"
                break

            if stop_reason == "MAX_LLM_CALLS_REACHED":
                break

        # 3. Final Synthesis
        chunk_evidence_lines = [
            f"[{cid}]: {c.get('content_snippet', '')}" for cid, c in list(accumulated_chunks.items())[:6]
        ]
        synthesis_context = (
            f"BÁO CÁO KHỞI TẠO (PRIMER):\n{primer_text}\n\n"
            f"BẰNG CHỨNG THU THẬP TỪ CÁC VÒNG LẶP DRIFT:\n"
            + "\n".join(chunk_evidence_lines)
        )

        synth_sys = (
            "Bạn là bộ máy DRIFT Search kết hợp giữa góc nhìn vĩ mô và bằng chứng vi mô. "
            "Hãy tổng hợp câu trả lời chi tiết và toàn diện nhất dựa trên ngữ cảnh đã điều tra. "
            "Bắt buộc trích dẫn mã [Chunk_ID] ở các chi tiết cụ thể. "
            f"Nếu không có thông tin hợp lệ, hãy trả lời chính xác: '{self.REFUSAL_MESSAGE}'."
        )
        synth_user = f"Ngữ cảnh DRIFT:\n{synthesis_context}\n\nCâu hỏi: {query}\n\nTổng hợp hoàn chỉnh:"

        final_resp = self._call_llm(synth_sys, synth_user)
        if final_resp:
            total_llm_calls += 1
            final_answer = final_resp.strip()
        else:
            top_chunk_id = list(accumulated_chunks.keys())[0] if accumulated_chunks else "CHUNK"
            final_answer = (
                f"{primer_text}\n\n"
                f"Qua các bước đào sâu DRIFT ({depth} tầng), hệ thống xác định chi tiết căn cứ từ [{top_chunk_id}]: "
                f"{list(accumulated_chunks.values())[0].get('content_snippet', '') if accumulated_chunks else ''}"
            )

        citations = list(accumulated_chunks.values())[:6]

        trace = GraphRAGQueryTrace(
            query_id=query_id,
            query_text=query,
            query_mode="DRIFT",
            selected_chunk_ids=list(accumulated_chunks.keys()),
            drift_steps=drift_steps,
            stop_reason=stop_reason,
            llm_calls=total_llm_calls,
            latency_ms=round((time.perf_counter() - start_time) * 1000, 2),
            details={"thinking": self.last_thought} if self.last_thought else {},
        )

        return GraphRAGQueryResult(
            query_id=query_id,
            query_text=query,
            query_mode="DRIFT",
            answer=final_answer,
            grounded_citations=citations,
            trace=trace,
        )
