# LEGAL RETRIEVAL LAB
## Demo chứng minh hiệu quả của Chunking, Embedding và Knowledge Graph trong truy xuất văn bản quy phạm pháp luật Việt Nam

Dự án nghiên cứu và đánh giá thực nghiệm có kiểm chứng nhằm so sánh các kỹ thuật truy xuất thông tin pháp lý trên kho dữ liệu văn bản quy phạm pháp luật Việt Nam thực tế (**110+ Văn bản QPPL, 10.125 Điều/Khoản/Điểm, 10.870 quan hệ Knowledge Graph**).

---

## 1. Kiến trúc hệ thống

```text
d:\GitHub\test-kg-rag\
├── config/             # Cấu hình hệ thống (model embedding, top-k, chunk size, KG weights)
├── data/
│   ├── processed/      # 110+ VBQPPL chuẩn, quan hệ KG và bộ 60 câu hỏi Benchmark
│   └── mau_data.json   # Dữ liệu nguồn Bộ Pháp điển Việt Nam (Chủ đề 1: An ninh quốc gia)
├── storage/            # SQLite Database Manager (idempotent CRUD, hash check) & Data Models
├── ingestion/          # Bộ chuyển đổi Pháp điển -> VBQPPL & Structure Parser (Điều -> Khoản -> Điểm)
├── chunking/           # 3 chiến lược: A1 (Article), A2 (Fixed Token), A3 (Structure-aware)
├── embedding/          # Adapter SentenceTransformers (GPU CUDA / fallback CPU) & FAISS Index Manager
├── retrieval/          # B1 (BM25), B2 (Dense Vector), B3 (Hybrid Reciprocal Rank Fusion - RRF)
├── knowledge_graph/    # Đồ thị NetworkX, trích xuất dẫn chiếu & mở rộng đồ thị 1-hop / 2-hop
├── temporal/           # Bộ lọc hiệu lực theo ngày tham chiếu & lịch sử sửa đổi (ProvisionVersion)
├── evaluation/         # Bộ tính metrics (Recall, MRR, nDCG, HitRate, KG Precision) & Benchmark Runner
├── app/                # Giao diện trực quan Streamlit 4 Tabs
├── scripts/            # Script chuẩn bị dữ liệu (prepare) và chạy benchmark CLI
└── tests/              # Bộ 12 test cases tự động (pytest) theo Mục 14 plan_demo.md
```

---

## 2. Các thành phần thí nghiệm

### Thí nghiệm A: So sánh 3 chiến lược Chunking
1. **A1 (Article-level Chunking):** Mỗi Điều là một đơn vị truy xuất, giữ trọn vẹn ngữ cảnh. Các Điều vượt quá 512 tokens được phân tách tại ranh giới Khoản để không bị cắt xén nội dung.
2. **A2 (Fixed-size Token Chunking):** Cắt theo cửa sổ cố định 256 tokens, overlap 32 tokens. Ghi nhận và gắn cờ các chunk bị cắt ngang qua ranh giới Khoản/Điểm (`crosses_boundary=True`).
3. **A3 (Structure-aware Legal Chunking):** Chia theo đơn vị Điều/Khoản/Điểm nhỏ nhất có nghĩa, đồng thời **tiêm tiêu đề văn bản, tên Chương, tên Điều và câu dẫn cha (Context Header Injection)** vào đầu mỗi chunk.

### Thí nghiệm B: So sánh 3 chế độ Retrieval
1. **B1 (BM25):** Tìm kiếm từ khóa chính xác, bảo toàn số hiệu văn bản (`32/2004/QH11`, `112/2014/NĐ-CP`) và cụm từ pháp lý.
2. **B2 (Dense Retrieval):** Tìm kiếm ngữ nghĩa vector bằng `bkai-foundation-models/vietnamese-bi-encoder` (chạy trên GPU CUDA / fallback CPU) và chỉ mục FAISS `IndexFlatIP`.
3. **B3 (Hybrid Retrieval):** Kết hợp thứ hạng BM25 và Dense bằng thuật toán **Reciprocal Rank Fusion (RRF)**:
   $$\text{Score}_{\text{RRF}}(d) = \frac{0.5}{60 + \text{rank}_{\text{bm25}}} + \frac{0.5}{60 + \text{rank}_{\text{dense}}}$$

### Thí nghiệm C: Knowledge Graph & Temporal Filtering
1. **Căn cứ đồ thị:** Được xây dựng từ 551 liên kết `Chỉ dẫn` chính thức của Bộ Tư pháp, 35 điều khoản có lịch sử sửa đổi/bổ sung (`AMENDS`, `SUPPLEMENTS`, `REPEALS`), và quan hệ quy định chi tiết thi hành (`DETAILS`).
2. **Graph Expansion:** Từ các kết quả seed ban đầu, mở rộng 1-hop / 2-hop theo các loại quan hệ được chọn, chấm điểm suy giảm theo khoảng cách hop và mức độ trùng khớp từ khóa, lưu trữ minh bạch đường đi (`expansion_path`) và căn cứ chứng minh (`evidence_text`).
3. **Kiểm soát hiệu lực (Temporal):** Kiểm tra ngày tham chiếu (`reference_date`) cho cả seed nodes và expanded nodes. Hỗ trợ tra cứu lịch sử phiên bản (`ProvisionVersion v1` trước sửa đổi và `v2` sau sửa đổi).

---

## 3. Hướng dẫn cài đặt & Khởi chạy

### Bước 1: Cài đặt thư viện phụ thuộc
Hệ thống sử dụng Python 3.10+ (đã kiểm thử trên Python 3.13):
```bash
pip install -r requirements.txt
```

### Bước 2: Chuẩn bị dữ liệu & Lập chỉ mục toàn bộ
Chạy script chuẩn bị để chuyển đổi dữ liệu từ `mau_data.json`, nạp vào SQLite, phân đoạn theo A1/A2/A3, tạo chỉ mục BM25, tạo vector embedding FAISS và sinh 60 câu hỏi Benchmark:
```bash
python scripts/prepare_data_and_indices.py
```

### Bước 3: Chạy giao diện demo Streamlit
Khởi chạy ứng dụng giao diện trực quan 4 tabs:
```bash
streamlit run app/main.py
```
Truy cập trình duyệt tại: `http://localhost:8501`

---

## 4. Hướng dẫn chạy Benchmark & Kiểm thử

### Chạy Benchmark từ dòng lệnh:
Chạy đánh giá ma trận 18 cấu hình trên bộ 60 câu hỏi (phân bổ đều 5 nhóm):
```bash
# Chạy toàn bộ 60 câu hỏi:
python scripts/run_benchmark_cli.py

# Hoặc chạy thử nghiệm nhanh trên mẫu 10 câu hỏi:
python scripts/run_benchmark_cli.py 10
```
Báo cáo kết quả sẽ được tự động xuất ra:
- `artifacts/reports/benchmark_summary.csv`
- `artifacts/reports/benchmark_by_group.csv`
- `artifacts/reports/benchmark_full.json`

### Chạy kiểm thử tự động (pytest):
Bộ 12 test cases bao phủ toàn diện các yêu cầu chất lượng của `plan_demo.md`:
```bash
pytest tests/test_pipeline.py -v
```

---

## 5. Bảng điều khiển Giao diện (Streamlit Tabs)

1. **Tab 0 — 🌐 Platform Architecture:** Thanh tra trực quan Pipeline Processing Profile Registry, xem JSON schema của bộ 7 Canonical Models, thử nghiệm Validator bắt lỗi tự động và chạy thử Sandbox toàn trình.
2. **Tab 1 — 🧭 Multi-Domain Lab:** Thử nghiệm GraphRAG trên các domain tổng hợp (`synth_domain_ops`, `synth_domain_rd`), cách ly đồ thị theo namespace, lọc bảo mật theo quyền người dùng ACL Policy và thanh tra nguồn gốc bằng chứng Provenance Inspector.
3. **Tab 2 — 🔍 Interactive Search (Legal):** Tra cứu câu hỏi pháp luật, bật/tắt KG Expansion, chọn độ sâu 1/2 hops, chọn ngày tham chiếu và xem kết quả đối chiếu có/không có KG kèm kiểm chứng trích dẫn LLM Generator.
4. **Tab 3 — 🔬 Chunking Lab (Legal):** Xem trực quan ranh giới cắt của A1 vs A2 vs A3, bảng thống kê số token, tỷ lệ cắt ngang ranh giới và so sánh kết quả retrieval trên cùng câu hỏi.
5. **Tab 4 — 🕸️ Knowledge Graph (Legal):** Khám phá mạng lưới node/edge pháp điển, xem quan hệ trỏ vào/trỏ ra của từng điều khoản và dòng thời gian lịch sử phiên bản sửa đổi.
6. **Tab 5 — 📊 Benchmark & Evaluation (Legal):** Xem bảng xếp hạng 18 cấu hình, biểu đồ so sánh Recall/MRR và phân rã hiệu quả theo 5 nhóm câu hỏi pháp lý.

---

## 6. KIẾN TRÚC NỀN TẢNG MỞ RỘNG (EXTENSIBLE MULTI-DOMAIN GRAPHRAG PLATFORM)

Nền tảng được thiết kế theo kiến trúc module hoá cắm ghép (pluggable), cho phép tiếp nhận bất kỳ domain tri thức hoặc định dạng tài liệu mới nào mà **không cần sửa đổi mã nguồn cốt lõi**.

### 6.1. Sơ đồ Luồng Xử Lý Tổng Quát (Pipeline Architecture)
```text
Raw Text / Document
       │
       ▼
[Domain & Profile Resolution] ──→ (Tìm Profile phù hợp từ ProfileRegistry hoặc fallback)
       │
       ▼
[IDocumentParser]             ──→ Bóc tách thành danh sách ContentBlock (HEADING, PARAGRAPH, LIST_ITEM...)
       │
       ▼
[IChunker]                    ──→ Cắt thành CanonicalChunk, bảo toàn source_block_ids & SourceLocation
       │
       ▼
[IMetadata & IKnowledgeExtractor] ──→ Trích xuất CanonicalEntity, EntityMention, CanonicalRelationship, EvidenceRecord
       │
       ▼
[CanonicalSchemaValidator]    ──→ Kiểm tra tính toàn vẹn khoá ngoại, ranh giới domain, bằng chứng VERIFIED
       │
       ▼
[Storage & Namespace Indexing] ──→ Lưu vào SQLite (platform_*), dựng đồ thị NetworkX theo graph_namespace,
                                   lập chỉ mục BM25 + FAISS Vector gắn với domain.
```

### 6.2. Bộ Canonical Metadata Schema
Bộ 7 mô hình dữ liệu chuẩn hoá độc lập với domain (định nghĩa tại `core/schema.py`):
1. **`CanonicalDocument`**: Thông tin định danh, domain, phiên bản profile, trạng thái xử lý, quyền truy cập `ACLPolicy` và metadata mở rộng.
2. **`ContentBlock`**: Đơn vị nội dung cấu trúc (HEADING, PARAGRAPH, CODE, TABLE...), lưu toạ độ offset tuyệt đối (`start_offset`, `end_offset`) và phân cấp cha-con (`parent_block_id`).
3. **`CanonicalChunk`**: Đơn vị lập chỉ mục và truy xuất, bắt buộc liên kết với ít nhất một `source_block_ids`, lưu trữ chiến lược chunking và phân loại bảo mật.
4. **`EvidenceRecord`**: Bản ghi chứng cứ trích dẫn (Provenance), lưu đoạn trích dẫn nguyên văn (`quote_text`), toạ độ vị trí, mã trích xuất (`extractor_id`) và nguồn gốc (`FIXTURE_PROVIDED` | `EXTRACTOR_GENERATED`).
5. **`CanonicalEntity`**: Thực thể tri thức chuẩn hoá `{domain_id}::ENT::{type}::{slug}`, gán nhãn `graph_namespace` và liên kết bằng chứng `evidence_refs`.
6. **`EntityMention`**: Vị trí xuất hiện cụ thể của thực thể trong từng chunk/block.
7. **`CanonicalRelationship`**: Quan hệ có hướng giữa hai thực thể, gán nhãn `graph_namespace`, độ tin cậy `confidence`, trạng thái xác minh (`VERIFIED` | `UNVERIFIED`) và bắt buộc liên kết `evidence_refs` hợp lệ khi đã xác minh.
8. **`ACLPolicy`**: Chính sách bảo mật kiểm soát quyền truy cập theo vai trò (`allowed_roles`), người dùng (`allowed_users`) và cấp độ bảo mật (`PUBLIC`, `INTERNAL`, `CONFIDENTIAL`, `RESTRICTED`).

### 6.3. Hướng Dẫn Từng Bước Mở Rộng Nền Tảng

#### Bước 1: Đăng ký một Domain mới
Thêm cấu hình vào file `config/platform_profiles.yaml`:
```yaml
domains:
  - domain_id: "medical_records"
    name: "Hồ sơ Bệnh án & Dược học"
    description: "Tài liệu y khoa, phác đồ điều trị, dược thư quốc gia."
    enabled: true
    schema_version: "1.0.0"
    graph_namespace: "ns_med"
    default_profile_id: "profile_med_standard"
    is_test_fixture: false
    allowed_entity_types: ["DISEASE", "DRUG", "SYMPTOM", "DOSAGE"]
    allowed_predicates: ["TREATS", "CAUSES", "CONTRAINDICATED_WITH"]
```

#### Bước 2: Định nghĩa một Processing Profile mới
```yaml
profiles:
  - profile_id: "profile_med_standard"
    domain_id: "medical_records"
    document_type: "clinical_guideline"
    profile_version: "1.0.0"
    parser_strategy: "generic_paragraph"
    chunking_strategy: "sliding_window"
    chunking_parameters:
      max_tokens: 256
      chunk_overlap: 32
    metadata_extractors:
      - "basic_structural"
    fallback_profile_id: "profile_generic_default"
    enabled: true
```

#### Bước 3: Viết một Chunker hoặc Parser mới
Kế thừa interface chuẩn từ `pipeline/interfaces.py` và đăng ký với Registry:
```python
from pipeline.interfaces import IChunker
from pipeline.chunkers import ChunkerRegistry

class SemanticMedicalChunker(IChunker):
    def chunk(self, doc, blocks, profile):
        # Triển khai thuật toán cắt theo ngữ nghĩa y khoa...
        return chunks

# Đăng ký chiến lược mới
ChunkerRegistry.register("semantic_medical", SemanticMedicalChunker())
```

#### Bước 4: Viết một Knowledge Extractor mới
Kế thừa `IKnowledgeExtractor` và trích xuất thực thể kèm `EvidenceRecord`:
```python
from pipeline.interfaces import IKnowledgeExtractor
from core.schema import CanonicalEntity, CanonicalRelationship, EvidenceRecord

class MedicalEntityExtractor(IKnowledgeExtractor):
    def extract_knowledge(self, doc, blocks, chunks, profile):
        # Trích xuất thực thể và luôn kèm EvidenceRecord chứng minh
        ev = EvidenceRecord(
            evidence_id=f"ev_{doc.document_id}_01",
            domain_id=doc.domain_id,
            document_id=doc.document_id,
            quote_text="Paracetamol chỉ định giảm đau, hạ sốt.",
            verification_status="VERIFIED",
            origin_type="EXTRACTOR_GENERATED",
        )
        # Tạo entities và relationships trỏ tới ev.evidence_id...
        return [entity], [mention], [relationship], [ev]
```

### 6.4. Quản lý Đồ thị Đa Miền & Kiểm soát Bảo mật (Multi-Graph Isolation & ACL)
- **Namespace Isolation:** Các truy vấn đồ thị luôn được ràng buộc trong `graph_namespace`. Node và Edge của domain A tuyệt đối không xuất hiện trong domain B.
- **Cross-Domain Mapping:** Chỉ khi bật cờ `allow_cross_domain=True`, các liên kết liên miền đã xác minh từ bảng `platform_entity_mappings` mới được kết nối vào đồ thị.
- **Candidate vs. Verified:** Hệ thống mặc định chỉ đưa các liên kết đã xác minh (`VERIFIED`) vào đồ thị và ngữ cảnh RAG. Các liên kết ứng viên (`UNVERIFIED`) chỉ được hiển thị khi bật cờ `include_candidates=True`.
- **ACL Policy Enforcement:** Tại tầng retrieval, mọi chunk đều được kiểm tra đồng thời:
  1. `security_classification`: Chặn các chunk `CONFIDENTIAL` / `RESTRICTED` nếu người dùng chỉ có quyền `PUBLIC` / `INTERNAL`.
  2. `allowed_roles`: Người dùng bắt buộc phải có vai trò nằm trong danh sách `allowed_roles` của chunk.

### 6.5. Hướng Dẫn Chạy Kiểm Thử & Thẩm Định Nền Tảng (Platform Verifier)
Nền tảng tích hợp bộ kiểm thử tự động 11 tiêu chí chất lượng:
```bash
# Nạp dữ liệu synthetic test fixtures và lập chỉ mục đa miền:
python scripts/seed_platform_fixtures.py

# Chạy bộ thẩm định tự động 11 tiêu chí nền tảng:
python evaluation/platform_verifier.py

# Chạy toàn bộ 42 unit & integration tests:
pytest -v
```

Báo cáo kết quả thẩm định được lưu tại `artifacts/reports/platform_verification_report.json`.

---

## 7. Ghi Chú Ranh Giới Dữ Liệu (Synthetic Fixtures vs. Customer Benchmark)
> [!IMPORTANT]
> - Các domain mẫu `synth_domain_ops` và `synth_domain_rd` trong thư mục `data/fixtures/` là **dữ liệu tổng hợp (Synthetic Test Fixtures)** được tạo ra nhằm mục đích kiểm tra kỹ thuật tính toàn vẹn của nền tảng, cơ chế cách ly namespace và kiểm soát ACL.
> - **KHÔNG ĐƯỢC COI** kết quả đánh giá trên dữ liệu tổng hợp này là số liệu đo lường hiệu năng trên dữ liệu thực tế của khách hàng.
> - Khi triển khai cho khách hàng mới, chỉ cần khai báo file cấu hình domain/profile tương ứng và đưa tài liệu qua `PipelineOrchestrator` mà không cần thay đổi bất kỳ logic nền tảng nào.

