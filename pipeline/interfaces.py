from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Tuple, Optional
import numpy as np

from core.schema import (
    CanonicalDocument,
    ContentBlock,
    CanonicalChunk,
    EvidenceRecord,
    CanonicalEntity,
    EntityMention,
    CanonicalRelationship,
)
from core.registry import ProcessingProfile


class IDocumentParser(ABC):
    @abstractmethod
    def parse(
        self, raw_text: str, doc: CanonicalDocument, config: Optional[Dict[str, Any]] = None
    ) -> List[ContentBlock]:
        """Bóc tách văn bản thô thành danh sách ContentBlock chuẩn."""
        pass


class IChunker(ABC):
    @abstractmethod
    def chunk(
        self,
        doc: CanonicalDocument,
        blocks: List[ContentBlock],
        profile: ProcessingProfile,
    ) -> List[CanonicalChunk]:
        """Tạo các CanonicalChunk phục vụ retrieval từ Document và ContentBlocks."""
        pass


class IMetadataExtractor(ABC):
    @abstractmethod
    def extract_metadata(
        self,
        doc: CanonicalDocument,
        blocks: List[ContentBlock],
        config: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Trích xuất metadata tài liệu từ cấu trúc hoặc nội dung."""
        pass


class IKnowledgeExtractor(ABC):
    @abstractmethod
    def extract_knowledge(
        self,
        doc: CanonicalDocument,
        blocks: List[ContentBlock],
        chunks: List[CanonicalChunk],
        profile: ProcessingProfile,
    ) -> Tuple[
        List[CanonicalEntity],
        List[EntityMention],
        List[CanonicalRelationship],
        List[EvidenceRecord],
    ]:
        """Trích xuất Entities, Mentions, Relationships và Evidences."""
        pass


class ILocalModelAdapter(ABC):
    @abstractmethod
    def encode(self, texts: List[str]) -> np.ndarray:
        """Chuyển đổi danh sách văn bản thành ma trận vector embeddings."""
        pass

    @abstractmethod
    def get_dimension(self) -> int:
        """Kích thước vector chiều không gian embedding."""
        pass

    @abstractmethod
    def get_model_info(self) -> Dict[str, Any]:
        """Thông tin định danh và phiên bản của model."""
        pass
