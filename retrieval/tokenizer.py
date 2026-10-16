import re
from typing import List


class LegalTokenizer:
    DOC_NUM_RE = re.compile(r"\b\d+[a-z]?/\d{4}/[a-zđ\-]+", re.IGNORECASE)
    PROV_UNIT_RE = re.compile(r"\b(điều|khoản|điểm)\s+(\d+[a-z]?|[a-zđ])\b", re.IGNORECASE)

    @classmethod
    def tokenize(cls, text: str) -> List[str]:
        if not text:
            return []

        clean_text = text.lower()
        
        # Preserve document numbers by replacing slashes with underscores temporarily
        doc_matches = cls.DOC_NUM_RE.findall(clean_text)
        for dm in doc_matches:
            rep = dm.replace("/", "_").replace("-", "_")
            clean_text = clean_text.replace(dm, f" {rep} ")

        # Preserve units like "điều 1" -> "điều_1"
        clean_text = cls.PROV_UNIT_RE.sub(r"\1_\2", clean_text)

        # Standard word tokenization
        tokens = re.findall(r"\w+", clean_text, re.UNICODE)
        return tokens
