import re
from typing import List, Dict, Any, Optional
from core.schema import CanonicalDocument, ContentBlock, SourceLocation
from pipeline.interfaces import IDocumentParser


class GenericParagraphParser(IDocumentParser):
    def parse(
        self, raw_text: str, doc: CanonicalDocument, config: Optional[Dict[str, Any]] = None
    ) -> List[ContentBlock]:
        if not raw_text or not raw_text.strip():
            return []

        blocks: List[ContentBlock] = []
        # Ensure markdown headings are separated into their own blocks
        normalized = re.sub(r"(?m)^(#{1,6}\s+.*)$", r"\n\n\1\n\n", raw_text)
        pattern = re.compile(r"(?:\r?\n\s*){2,}")
        segments = pattern.split(normalized)

        curr_pos = 0
        block_idx = 0

        for seg in segments:
            seg_stripped = seg.strip()
            if not seg_stripped:
                continue

            start_offset = raw_text.find(seg_stripped, curr_pos)
            if start_offset == -1:
                start_offset = curr_pos
            end_offset = start_offset + len(seg_stripped)
            curr_pos = end_offset

            # Check if this single line paragraph looks like a heading
            lines = seg_stripped.splitlines()
            is_heading = len(lines) == 1 and (
                seg_stripped.startswith("#")
                or re.match(r"^(?:Chương|Mục|Phần|Điều|\d+[\.\)])\s+", seg_stripped, re.I)
            )

            b_type = "HEADING" if is_heading else "PARAGRAPH"
            b_id = f"{doc.document_id}__BLK_{block_idx:04d}"

            blocks.append(
                ContentBlock(
                    block_id=b_id,
                    document_id=doc.document_id,
                    domain_id=doc.domain_id,
                    block_index=block_idx,
                    block_type=b_type,
                    text=seg_stripped,
                    parent_block_id=None,
                    start_offset=start_offset,
                    end_offset=end_offset,
                    source_location=SourceLocation(
                        start_offset=start_offset,
                        end_offset=end_offset,
                        section_path=seg_stripped[:40] if is_heading else None,
                    ),
                )
            )
            block_idx += 1

        return blocks


class StructuredHeadingParser(IDocumentParser):
    """
    Parser nhận diện cấu trúc phân cấp (Heading/Sections)
    và tự động gán parent_block_id cho các đoạn văn con.
    """
    HEADING_RE = re.compile(
        r"^(#{1,6}\s+|(?:\d+\.){1,4}\s+|Chương\s+[IVXLCDM\d]+\.?\s*|Điều\s+\d+\.?\s*)(.*)$",
        re.IGNORECASE,
    )

    def parse(
        self, raw_text: str, doc: CanonicalDocument, config: Optional[Dict[str, Any]] = None
    ) -> List[ContentBlock]:
        if not raw_text or not raw_text.strip():
            return []

        lines = raw_text.splitlines()
        blocks: List[ContentBlock] = []

        curr_heading_id: Optional[str] = None
        current_paragraph_lines: List[str] = []
        current_para_start = 0
        curr_offset = 0
        block_idx = 0

        def flush_paragraph():
            nonlocal block_idx, current_paragraph_lines
            if not current_paragraph_lines:
                return
            p_text = "\n".join(current_paragraph_lines).strip()
            current_paragraph_lines = []
            if not p_text:
                return

            p_start = raw_text.find(p_text, current_para_start)
            if p_start == -1:
                p_start = current_para_start
            p_end = p_start + len(p_text)

            b_id = f"{doc.document_id}__BLK_{block_idx:04d}"
            blocks.append(
                ContentBlock(
                    block_id=b_id,
                    document_id=doc.document_id,
                    domain_id=doc.domain_id,
                    block_index=block_idx,
                    block_type="PARAGRAPH",
                    text=p_text,
                    parent_block_id=curr_heading_id,
                    start_offset=p_start,
                    end_offset=p_end,
                    source_location=SourceLocation(start_offset=p_start, end_offset=p_end),
                )
            )
            block_idx += 1

        for line in lines:
            line_stripped = line.strip()
            if not line_stripped:
                # blank line triggers flush
                flush_paragraph()
                curr_offset += len(line) + 1
                current_para_start = curr_offset
                continue

            h_match = self.HEADING_RE.match(line_stripped)
            if h_match:
                # Flush pending paragraph before new heading
                flush_paragraph()

                h_start = raw_text.find(line_stripped, curr_offset)
                if h_start == -1:
                    h_start = curr_offset
                h_end = h_start + len(line_stripped)

                h_id = f"{doc.document_id}__BLK_{block_idx:04d}"
                blocks.append(
                    ContentBlock(
                        block_id=h_id,
                        document_id=doc.document_id,
                        domain_id=doc.domain_id,
                        block_index=block_idx,
                        block_type="HEADING",
                        text=line_stripped,
                        parent_block_id=None,
                        start_offset=h_start,
                        end_offset=h_end,
                        source_location=SourceLocation(
                            start_offset=h_start,
                            end_offset=h_end,
                            section_path=line_stripped[:50],
                        ),
                    )
                )
                curr_heading_id = h_id
                block_idx += 1
                curr_offset += len(line) + 1
                current_para_start = curr_offset
            else:
                if not current_paragraph_lines:
                    current_para_start = curr_offset
                current_paragraph_lines.append(line)
                curr_offset += len(line) + 1

        flush_paragraph()
        return blocks


class ParserRegistry:
    _parsers: Dict[str, IDocumentParser] = {
        "generic_paragraph": GenericParagraphParser(),
        "structured_heading": StructuredHeadingParser(),
    }

    @classmethod
    def register(cls, name: str, parser: IDocumentParser) -> None:
        cls._parsers[name] = parser

    @classmethod
    def get(cls, name: str) -> IDocumentParser:
        if name not in cls._parsers:
            # Fallback to generic_paragraph
            return cls._parsers["generic_paragraph"]
        return cls._parsers[name]
