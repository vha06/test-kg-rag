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

    @classmethod
    def list_strategies(cls) -> List[str]:
        return list(cls._parsers.keys())


import io


def extract_text_from_file(file_input: Any, filename: Optional[str] = None) -> str:
    """
    Hỗ trợ đọc nội dung văn bản từ các tệp .txt, .md, .docx, .pdf.
    Hỗ trợ cả đường dẫn tệp (file path), bytes hoặc file-like object (BytesIO).
    """
    fn = ""
    if filename:
        fn = filename.lower()
    elif isinstance(file_input, str):
        fn = file_input.lower()

    # Xử lý input bytes / BytesIO
    if isinstance(file_input, bytes):
        raw_bytes = file_input
    elif hasattr(file_input, "read"):
        raw_bytes = file_input.read()
    elif isinstance(file_input, str):
        raw_bytes = None
    else:
        raise ValueError(f"Unsupported file_input type: {type(file_input)}")

    if fn.endswith((".txt", ".md", ".json", ".csv", ".log")):
        if raw_bytes is not None:
            for enc in ("utf-8", "utf-8-sig", "latin-1"):
                try:
                    return raw_bytes.decode(enc)
                except UnicodeDecodeError:
                    continue
            return raw_bytes.decode("utf-8", errors="ignore")
        else:
            for enc in ("utf-8", "utf-8-sig", "latin-1"):
                try:
                    with open(file_input, "r", encoding=enc) as f:
                        return f.read()
                except UnicodeDecodeError:
                    continue
            with open(file_input, "r", errors="ignore") as f:
                return f.read()

    elif fn.endswith(".docx"):
        bio = io.BytesIO(raw_bytes) if raw_bytes is not None else file_input
        try:
            import docx
            doc = docx.Document(bio)
            return "\n\n".join([p.text for p in doc.paragraphs if p.text.strip()])
        except Exception:
            import zipfile
            import xml.etree.ElementTree as ET
            try:
                z_target = io.BytesIO(raw_bytes) if raw_bytes is not None else file_input
                with zipfile.ZipFile(z_target) as z:
                    xml_content = z.read("word/document.xml")
                    tree = ET.fromstring(xml_content)
                    texts = [node.text for node in tree.iter() if node.text]
                    return " ".join(texts)
            except Exception as e:
                raise RuntimeError(f"Cannot parse docx file: {e}")

    elif fn.endswith(".pdf"):
        bio = io.BytesIO(raw_bytes) if raw_bytes is not None else file_input
        try:
            from pypdf import PdfReader
            reader = PdfReader(bio)
            pages_text = [page.extract_text() or "" for page in reader.pages]
            return "\n\n".join(pages_text)
        except Exception:
            try:
                import fitz
                if raw_bytes is not None:
                    doc = fitz.open(stream=raw_bytes, filetype="pdf")
                else:
                    doc = fitz.open(file_input)
                pages_text = [page.get_text() for page in doc]
                return "\n\n".join(pages_text)
            except Exception as e:
                raise RuntimeError(f"Cannot parse pdf file: {e}")

    if raw_bytes is not None:
        return raw_bytes.decode("utf-8", errors="ignore")
    with open(file_input, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


