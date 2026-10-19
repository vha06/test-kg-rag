import re
from typing import List, Dict, Any, Optional, Tuple
from core.schema import CanonicalDocument, ContentBlock, CanonicalChunk, SourceLocation
from core.registry import ProcessingProfile
from pipeline.interfaces import IChunker


def count_tokens(text: str) -> int:
    tokens = re.findall(r"\w+|[^\w\s]", text, re.UNICODE)
    return len(tokens)


class GenericSlidingWindowChunker(IChunker):
    """
    Chunker cửa sổ trượt tổng quát: gom các ContentBlocks,
    cắt theo max_tokens và chunk_overlap từ profile,
    bảo toàn danh sách source_block_ids.
    """

    def chunk(
        self,
        doc: CanonicalDocument,
        blocks: List[ContentBlock],
        profile: ProcessingProfile,
    ) -> List[CanonicalChunk]:
        if not blocks:
            return []

        params = profile.chunking_parameters or {}
        max_tokens = params.get("max_tokens", 256)
        overlap = params.get("chunk_overlap", 32)
        if overlap >= max_tokens:
            overlap = max(0, max_tokens // 4)

        chunks: List[CanonicalChunk] = []
        chunk_idx = 0

        # Build tokenized words per block
        tokenized_blocks = []
        for b in blocks:
            words = re.findall(r"\S+", b.text)
            tokenized_blocks.append((b, words))

        # Flatten words with block attribution
        word_stream = []  # list of (word_str, block_id, start_offset, end_offset)
        for b, words in tokenized_blocks:
            for w in words:
                word_stream.append((w, b.block_id, b.start_offset, b.end_offset))

        if not word_stream:
            return []

        step = max(1, max_tokens - overlap)
        curr_idx = 0

        while curr_idx < len(word_stream):
            window = word_stream[curr_idx : curr_idx + max_tokens]
            chunk_text = " ".join(item[0] for item in window)
            source_block_ids = list(dict.fromkeys(item[1] for item in window))

            # Approximate offsets
            min_start = min((item[2] for item in window if item[2] is not None), default=None)
            max_end = max((item[3] for item in window if item[3] is not None), default=None)

            c_id = f"{doc.document_id}__CHK_{chunk_idx:04d}"
            chunks.append(
                CanonicalChunk(
                    chunk_id=c_id,
                    document_id=doc.document_id,
                    domain_id=doc.domain_id,
                    source_block_ids=source_block_ids,
                    content=chunk_text,
                    chunk_index=chunk_idx,
                    chunking_strategy="sliding_window",
                    profile_id=profile.profile_id,
                    profile_version=profile.profile_version,
                    token_count=count_tokens(chunk_text),
                    source_location=SourceLocation(start_offset=min_start, end_offset=max_end),
                    security_classification=doc.security_classification,
                    acl=doc.acl,
                    metadata={"window_size": len(window)},
                )
            )
            chunk_idx += 1
            curr_idx += step

        return chunks


class BlockPreservingChunker(IChunker):
    """
    Chunker bảo toàn cấu trúc block: gom các khối nhỏ lại với nhau,
    chỉ ngắt ranh giới khi vượt max_tokens hoặc gặp block mới cấp cao hơn.
    """

    def chunk(
        self,
        doc: CanonicalDocument,
        blocks: List[ContentBlock],
        profile: ProcessingProfile,
    ) -> List[CanonicalChunk]:
        if not blocks:
            return []

        params = profile.chunking_parameters or {}
        max_tokens = params.get("max_tokens", 384)

        chunks: List[CanonicalChunk] = []
        chunk_idx = 0

        current_block_ids: List[str] = []
        current_texts: List[str] = []
        current_tokens = 0
        min_start: Optional[int] = None
        max_end: Optional[int] = None

        def flush_chunk():
            nonlocal chunk_idx, current_block_ids, current_texts, current_tokens, min_start, max_end
            if not current_texts:
                return

            c_content = "\n".join(current_texts).strip()
            c_id = f"{doc.document_id}__CHK_{chunk_idx:04d}"
            chunks.append(
                CanonicalChunk(
                    chunk_id=c_id,
                    document_id=doc.document_id,
                    domain_id=doc.domain_id,
                    source_block_ids=list(dict.fromkeys(current_block_ids)),
                    content=c_content,
                    chunk_index=chunk_idx,
                    chunking_strategy="block_preserving",
                    profile_id=profile.profile_id,
                    profile_version=profile.profile_version,
                    token_count=count_tokens(c_content),
                    source_location=SourceLocation(start_offset=min_start, end_offset=max_end),
                    security_classification=doc.security_classification,
                    acl=doc.acl,
                )
            )
            chunk_idx += 1
            current_block_ids = []
            current_texts = []
            current_tokens = 0
            min_start = None
            max_end = None

        for b in blocks:
            b_text = b.text.strip()
            if not b_text:
                continue

            b_tokens = count_tokens(b_text)

            # If adding this block exceeds max_tokens and we already have accumulated blocks, flush first
            if current_tokens + b_tokens > max_tokens and current_texts:
                flush_chunk()

            # If a single block alone is larger than max_tokens, split it or add as its own chunk
            if b_tokens > max_tokens and not current_texts:
                # Add as standalone chunk
                c_id = f"{doc.document_id}__CHK_{chunk_idx:04d}"
                chunks.append(
                    CanonicalChunk(
                        chunk_id=c_id,
                        document_id=doc.document_id,
                        domain_id=doc.domain_id,
                        source_block_ids=[b.block_id],
                        content=b_text,
                        chunk_index=chunk_idx,
                        chunking_strategy="block_preserving",
                        profile_id=profile.profile_id,
                        profile_version=profile.profile_version,
                        token_count=b_tokens,
                        source_location=b.source_location,
                        security_classification=doc.security_classification,
                        acl=doc.acl,
                    )
                )
                chunk_idx += 1
                continue

            current_block_ids.append(b.block_id)
            current_texts.append(b_text)
            current_tokens += b_tokens
            if min_start is None or (b.start_offset is not None and b.start_offset < min_start):
                min_start = b.start_offset
            if max_end is None or (b.end_offset is not None and b.end_offset > max_end):
                max_end = b.end_offset

        flush_chunk()
        return chunks


class HierarchicalHeadingChunker(IChunker):
    """
    Chunker phân cấp theo tiêu đề mục:
    Gom các khối văn bản (PARAGRAPH) thuộc cùng một tiêu đề cha (HEADING).
    Đính kèm tiêu đề ngữ cảnh vào đầu mỗi chunk để tránh mất ngữ cảnh khi cắt nhỏ.
    """

    def chunk(
        self,
        doc: CanonicalDocument,
        blocks: List[ContentBlock],
        profile: ProcessingProfile,
    ) -> List[CanonicalChunk]:
        if not blocks:
            return []

        params = profile.chunking_parameters or {}
        max_tokens = params.get("max_tokens", 300)

        chunks: List[CanonicalChunk] = []
        chunk_idx = 0

        # Group blocks by heading
        # A section is: (heading_block, [child_blocks])
        sections: List[Tuple[Optional[ContentBlock], List[ContentBlock]]] = []
        curr_heading: Optional[ContentBlock] = None
        curr_children: List[ContentBlock] = []

        for b in blocks:
            if b.block_type == "HEADING":
                if curr_heading is not None or curr_children:
                    sections.append((curr_heading, curr_children))
                curr_heading = b
                curr_children = []
            else:
                curr_children.append(b)
        if curr_heading is not None or curr_children:
            sections.append((curr_heading, curr_children))

        for heading, children in sections:
            prefix = f"[{heading.text.strip()}]\n" if heading else ""
            prefix_tokens = count_tokens(prefix)
            available_tokens = max(50, max_tokens - prefix_tokens)

            # Accumulate children into chunks
            curr_chunk_blocks: List[ContentBlock] = [heading] if heading else []
            curr_chunk_texts: List[str] = [prefix] if prefix else []
            curr_tokens = prefix_tokens
            min_start = heading.start_offset if heading else None
            max_end = heading.end_offset if heading else None

            def flush_section_chunk():
                nonlocal chunk_idx, curr_chunk_blocks, curr_chunk_texts, curr_tokens, min_start, max_end
                if not curr_chunk_texts or (prefix and len(curr_chunk_texts) == 1 and curr_chunk_texts[0] == prefix):
                    return

                c_content = "".join(curr_chunk_texts).strip()
                b_ids = [b.block_id for b in curr_chunk_blocks if b]
                c_id = f"{doc.document_id}__CHK_{chunk_idx:04d}"
                chunks.append(
                    CanonicalChunk(
                        chunk_id=c_id,
                        document_id=doc.document_id,
                        domain_id=doc.domain_id,
                        source_block_ids=list(dict.fromkeys(b_ids)),
                        content=c_content,
                        chunk_index=chunk_idx,
                        chunking_strategy="hierarchical_heading",
                        profile_id=profile.profile_id,
                        profile_version=profile.profile_version,
                        token_count=count_tokens(c_content),
                        source_location=SourceLocation(start_offset=min_start, end_offset=max_end),
                        security_classification=doc.security_classification,
                        acl=doc.acl,
                        metadata={"section_heading": heading.text if heading else None},
                    )
                )
                chunk_idx += 1
                curr_chunk_blocks = [heading] if heading else []
                curr_chunk_texts = [prefix] if prefix else []
                curr_tokens = prefix_tokens
                min_start = heading.start_offset if heading else None
                max_end = heading.end_offset if heading else None

            for child in children:
                ch_text = child.text.strip()
                if not ch_text:
                    continue
                ch_tokens = count_tokens(ch_text)

                if curr_tokens + ch_tokens > max_tokens and len(curr_chunk_texts) > (1 if prefix else 0):
                    flush_section_chunk()

                curr_chunk_blocks.append(child)
                curr_chunk_texts.append(ch_text + "\n\n")
                curr_tokens += ch_tokens
                if min_start is None or (child.start_offset is not None and child.start_offset < min_start):
                    min_start = child.start_offset
                if max_end is None or (child.end_offset is not None and child.end_offset > max_end):
                    max_end = child.end_offset

            flush_section_chunk()

        return chunks


class FixedTokenChunker(IChunker):
    """
    Chunker cắt cố định theo số lượng token:
    Cắt chuỗi từ vựng thành các đoạn đúng kích cỡ max_tokens,
    bảo toàn danh sách source_block_ids và tọa độ SourceLocation.
    """

    def chunk(
        self,
        doc: CanonicalDocument,
        blocks: List[ContentBlock],
        profile: ProcessingProfile,
    ) -> List[CanonicalChunk]:
        if not blocks:
            return []

        params = profile.chunking_parameters or {}
        max_tokens = params.get("max_tokens", 256)
        overlap = params.get("chunk_overlap", 0)
        if overlap >= max_tokens:
            overlap = max(0, max_tokens // 4)

        # Delegate to sliding window logic with exact overlap
        sliding = GenericSlidingWindowChunker()
        chunks = sliding.chunk(doc, blocks, profile)
        for c in chunks:
            c.chunking_strategy = "fixed_token"
        return chunks


class ChunkerRegistry:
    _chunkers: Dict[str, IChunker] = {
        "sliding_window": GenericSlidingWindowChunker(),
        "block_preserving": BlockPreservingChunker(),
        "hierarchical_heading": HierarchicalHeadingChunker(),
        "fixed_token": FixedTokenChunker(),
    }

    @classmethod
    def register(cls, name: str, chunker: IChunker) -> None:
        cls._chunkers[name] = chunker

    @classmethod
    def get(cls, name: str) -> IChunker:
        if name not in cls._chunkers:
            return cls._chunkers["sliding_window"]
        return cls._chunkers[name]

    @classmethod
    def list_strategies(cls) -> List[str]:
        return list(cls._chunkers.keys())

