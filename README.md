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

## 5. Bảng điều khiển Giao diện (Streamlit 4 Tabs)

1. **Tab 1 — Interactive Search:** Tra cứu câu hỏi tự do hoặc chọn câu hỏi mẫu, bật/tắt KG Expansion, chọn độ sâu 1/2 hops, chọn ngày tham chiếu và xem kết quả kèm thẻ nguồn, đường đi giải thích KG.
2. **Tab 2 — Chunking Lab:** Chọn một văn bản/điều khoản bất kỳ để xem trực quan ranh giới cắt của A1 vs A2 vs A3, bảng thống kê số token, tỷ lệ cắt ngang ranh giới và so sánh kết quả retrieval trên cùng câu hỏi.
3. **Tab 3 — Knowledge Graph Explorer:** Khám phá mạng lưới node/edge, xem chi tiết quan hệ trỏ vào/trỏ ra của từng điều khoản và dòng thời gian lịch sử phiên bản sửa đổi.
4. **Tab 4 — Benchmark & Evaluation:** Bấm nút chạy benchmark trực tiếp, xem bảng xếp hạng 18 cấu hình, biểu đồ so sánh Recall/MRR và phân rã hiệu quả theo 5 nhóm câu hỏi pháp lý.
