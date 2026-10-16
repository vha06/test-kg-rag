# PROJECT PLAN: LEGAL RETRIEVAL LAB
## Demo chứng minh hiệu quả của Chunking, Embedding và Knowledge Graph trong truy xuất văn bản quy phạm pháp luật Việt Nam

## 1. MỤC TIÊU DỰ ÁN

Xây dựng một demo có giao diện trực quan để đánh giá và so sánh hiệu quả của các kỹ thuật truy xuất dữ liệu trên một kho văn bản quy phạm pháp luật Việt Nam có quy mô lớn.

Demo tập trung chứng minh ba yếu tố:

1. **Chunking:** Cách chia văn bản thành các đơn vị truy xuất ảnh hưởng như thế nào đến khả năng tìm được đúng điều, khoản, điểm và giữ đầy đủ ngữ cảnh pháp lý.
2. **Embedding:** Tìm kiếm ngữ nghĩa bằng vector có lợi ích gì so với tìm kiếm từ khóa; khi kết hợp với BM25, chất lượng truy xuất thay đổi như thế nào.
3. **Knowledge Graph:** Việc khai thác quan hệ giữa các văn bản và điều khoản có giúp tìm được những căn cứ pháp lý liên quan mà tìm kiếm thông thường bỏ sót hay không.

Mục tiêu cuối cùng là tạo ra một môi trường thí nghiệm có thể kiểm chứng, trong đó người dùng chạy cùng một câu hỏi qua nhiều cấu hình, xem kết quả truy xuất, quan hệ giữa các điều khoản và các chỉ số đánh giá.

Không được mặc định rằng phương pháp phức tạp hơn sẽ tốt hơn. Mọi kết luận về hiệu quả phải dựa trên benchmark thực tế.

## 2. PHẠM VI TRIỂN KHAI

### Trong phạm vi

- Đọc và chuẩn hóa dữ liệu văn bản quy phạm pháp luật tiếng Việt.
- Xây dựng nhiều chiến lược chunking có thể cấu hình.
- Tạo, lưu trữ và tái sử dụng embedding.
- Triển khai BM25, dense retrieval và hybrid retrieval.
- Xây dựng Knowledge Graph biểu diễn cấu trúc và quan hệ pháp lý.
- Cho phép bật/tắt Knowledge Graph và điều chỉnh phạm vi mở rộng.
- Xây dựng benchmark có đáp án chuẩn, tính toán các chỉ số retrieval.
- Xây dựng giao diện cho phép tìm kiếm, so sánh kết quả và trực quan hóa đồ thị.
- Lưu metadata và nguồn để truy xuất về đúng văn bản, điều khoản và phiên bản.

### Ngoài phạm vi của phiên bản đầu tiên

- Xây dựng chatbot tư vấn pháp luật hoàn chỉnh.
- Tự động đưa ra kết luận pháp lý có giá trị thay cho chuyên gia.
- Fine-tune embedding model ngay từ đầu.
- Xây dựng hệ thống Multi-Agent phức tạp.
- Dùng LLM tự động tạo mọi quan hệ pháp lý mà không có cơ chế xác minh.
- Triển khai hạ tầng phân tán hoặc graph database production nếu chưa cần thiết.

Ưu tiên hoàn thành một pipeline truy xuất có thể đo lường và giải thích trước. Có thể bổ sung LLM generation sau khi chất lượng retrieval đã được kiểm chứng.

## 3. KIỂM TRA WORKSPACE VÀ DỮ LIỆU TRƯỚC KHI CODE

Trước tiên, hãy kiểm tra repository hoặc workspace hiện tại:

- Xác định mã nguồn, dữ liệu, cấu trúc thư mục và môi trường đã có.
- Tìm hiểu định dạng dữ liệu, chất lượng metadata, cách định danh văn bản và quan hệ giữa các điều khoản.
- Kiểm tra xem đã có dữ liệu pháp luật hoặc pipeline chunking/embedding nào có thể tái sử dụng hay chưa.
- Không xóa hoặc ghi đè dữ liệu, chỉ mục hay mã nguồn hiện tại khi chưa có lý do rõ ràng.
- Nếu thiếu dữ liệu, hỗ trợ nhập dữ liệu từ file JSON, JSONL hoặc các định dạng văn bản phù hợp. Có thể bổ sung bộ dữ liệu mẫu nhỏ để kiểm thử chức năng.
- Ưu tiên nguồn pháp luật chính thức và ghi lại nguồn gốc, ngày tải, phiên bản dữ liệu và giấy phép sử dụng.
- Không tự tạo số hiệu văn bản, nội dung pháp luật hoặc quan hệ pháp lý giả rồi trình bày như dữ liệu thật.

Nếu dữ liệu thật chưa sẵn sàng, hãy tách riêng dữ liệu minh họa tổng hợp khỏi dữ liệu pháp luật đã xác minh. Không được báo cáo benchmark trên dữ liệu minh họa như kết quả đánh giá pháp luật thực tế.

## 4. KIẾN TRÚC TỔNG THỂ

Xây dựng pipeline gồm các tầng:

1. Data ingestion và chuẩn hóa văn bản.
2. Structure-aware parsing để xác định văn bản, chương, mục, điều, khoản và điểm.
3. Chunking với nhiều chiến lược có thể thay đổi.
4. Embedding và lập chỉ mục vector.
5. BM25 indexing.
6. Hybrid retrieval và xếp hạng kết quả.
7. Knowledge Graph construction và graph expansion.
8. Temporal metadata và lọc phiên bản khi có dữ liệu.
9. Benchmark và evaluation.
10. Giao diện tương tác để so sánh và giải thích kết quả.

Kiến trúc phải tách các module độc lập, để mỗi thí nghiệm chỉ cần thay đổi một thành phần mà không phải viết lại toàn bộ pipeline.

Stack mặc định đề xuất:

- Python làm ngôn ngữ xử lý dữ liệu.
- Streamlit cho giao diện demo.
- ChromaDB hoặc FAISS cho vector retrieval.
- BM25 cho keyword retrieval.
- SQLite hoặc PostgreSQL để lưu văn bản, metadata, phiên bản và quan hệ.
- NetworkX hoặc một graph layer tương đương để duyệt và trực quan hóa đồ thị.
- Một embedding model đa ngôn ngữ hoặc phù hợp với tiếng Việt, được cấu hình thay vì hard-code trong mã nguồn.

Chỉ bổ sung các thư viện hoặc dịch vụ khác khi có nhu cầu kỹ thuật cụ thể.

## 5. DATA INGESTION VÀ DATA MODEL

### 5.1. Chuẩn hóa văn bản

Mỗi văn bản nên có các metadata cơ bản:

- document_id ổn định.
- Số hiệu và tên văn bản.
- Loại văn bản.
- Cơ quan ban hành.
- Ngày ban hành.
- Ngày công bố hoặc đăng tải, nếu xác định được.
- Ngày bắt đầu hiệu lực và ngày kết thúc hiệu lực, nếu xác định được.
- Tình trạng hiệu lực theo nguồn dữ liệu.
- Source URL và thông tin phiên bản nguồn.

Cấu trúc văn bản cần được giữ lại đến cấp Điều, Khoản, Điểm nếu dữ liệu nguồn cho phép.

Mỗi điều khoản cần có định danh ổn định, văn bản cha, vị trí trong cây cấu trúc, nội dung nguyên gốc và liên kết đến nguồn.

### 5.2. Quản lý chunk

Mỗi chunk cần chứa:

- chunk_id.
- document_id.
- provision_id hoặc danh sách điều khoản nguồn.
- Chunking strategy.
- Nội dung chunk.
- Số token.
- Đường dẫn cấu trúc: Chương → Điều → Khoản → Điểm.
- Metadata pháp lý liên quan.
- Source URL.
- Metadata phiên bản và thời gian nếu có.

Không được chỉ lưu nội dung chunk mà mất liên kết đến điều khoản gốc. Việc truy xuất phải có khả năng hiển thị chính xác văn bản và vị trí nguồn.

### 5.3. Lưu trữ và cập nhật

- Chỉ mục phải được lưu bền vững để không phải embedding lại toàn bộ dữ liệu sau mỗi lần chạy.
- Hỗ trợ tái lập chỉ mục khi thay đổi mô hình embedding hoặc chiến lược chunking.
- Có cơ chế phát hiện tài liệu mới, tài liệu đã thay đổi và tài liệu bị xóa.
- Lưu cấu hình, phiên bản embedding model và thông tin build của từng index.
- Xử lý tài liệu lỗi hoặc metadata không hợp lệ bằng thông báo rõ ràng, không âm thầm bỏ qua.

## 6. THÍ NGHIỆM A: SO SÁNH CHUNKING

Triển khai ít nhất ba chiến lược:

### A1. Article-level chunking

Mỗi Điều là một đơn vị truy xuất, giữ nguyên các Khoản và Điểm bên trong nếu phù hợp với giới hạn của embedding model.

Nếu một Điều quá dài, phải tách tại ranh giới cấu trúc hợp lý thay vì cắt mất nội dung.

### A2. Fixed-size chunking

Chia văn bản theo giới hạn token có thể cấu hình, sử dụng overlap có thể điều chỉnh.

Cần ghi nhận trường hợp ranh giới chunk cắt qua cấu trúc Điều/Khoản/Điểm.

### A3. Structure-aware legal chunking

Chia theo cấu trúc pháp lý, ưu tiên Điều, Khoản, Điểm và các ranh giới ngữ nghĩa.

Nếu một đơn vị quá dài thì mới chia nhỏ hơn, đồng thời bổ sung metadata để truy ngược về đơn vị pháp lý gốc.

### Yêu cầu so sánh

- Cho phép người dùng chọn chiến lược chunking.
- Hiển thị ví dụ trực quan về cách mỗi chiến lược chia cùng một văn bản.
- Hiển thị số lượng chunk, phân bố số token, số chunk quá dài và số chunk quá ngắn.
- So sánh khả năng truy xuất đúng điều khoản trên cùng một bộ câu hỏi.
- Không thay đổi embedding model hoặc phương thức retrieval trong thí nghiệm chỉ tập trung vào chunking.

Lưu ý: thay đổi chunking thường đòi hỏi xây lại chỉ mục embedding. Hãy triển khai pipeline build index theo chiến lược đã chọn và cache kết quả để giảm thời gian thử nghiệm.

## 7. THÍ NGHIỆM B: SO SÁNH EMBEDDING VÀ RETRIEVAL

Triển khai ba chế độ:

### B1. BM25

Tìm kiếm từ khóa và thuật ngữ chính xác. Bảo đảm xử lý được số hiệu văn bản, số Điều, thuật ngữ pháp lý và tiếng Việt.

### B2. Dense retrieval

Chuyển câu hỏi và chunk thành embedding, sau đó tìm các chunk có độ tương đồng vector cao.

Embedding model phải được cấu hình rõ ràng. Có thể bắt đầu với một mô hình đa ngôn ngữ phù hợp tiếng Việt, sau đó mới so sánh với mô hình chuyên biệt hoặc đã fine-tune cho miền pháp luật.

### B3. Hybrid retrieval

Kết hợp danh sách kết quả BM25 và dense retrieval bằng một phương pháp hợp nhất thứ hạng, chẳng hạn Reciprocal Rank Fusion (RRF), hoặc phương pháp khác có thể giải thích và đánh giá.

### Yêu cầu

- Có thể chuyển đổi giữa BM25, dense và hybrid.
- Cố định số lượng ứng viên và top-k để so sánh công bằng.
- Hiển thị nguồn, nội dung, thứ hạng và điểm tương ứng nếu có ý nghĩa.
- Có thể kiểm tra cùng câu hỏi trên từng chế độ.
- Lưu lại cấu hình và kết quả để benchmark có thể tái lập.
- Không mặc định dense retrieval tốt hơn BM25.

## 8. THÍ NGHIỆM C: KNOWLEDGE GRAPH

### 8.1. Mục đích

Knowledge Graph dùng để bổ sung các điều khoản liên quan đến kết quả retrieval ban đầu, đặc biệt khi câu hỏi cần theo dấu dẫn chiếu, quan hệ giữa các văn bản hoặc lịch sử sửa đổi.

Knowledge Graph phải được xây dựng từ dữ liệu có căn cứ. Không được suy ra rằng hai điều khoản có quan hệ pháp lý chỉ vì nội dung của chúng giống nhau về ngữ nghĩa.

### 8.2. Các loại node

Ít nhất cần có:

- LegalDocument: văn bản pháp luật.
- Provision: Điều, Khoản, Điểm.
- ProvisionVersion: phiên bản nội dung của điều khoản nếu có dữ liệu lịch sử.

Có thể bổ sung LegalEvent để mô hình hóa sự kiện sửa đổi, bổ sung, thay thế hoặc bãi bỏ.

### 8.3. Các loại edge

- CONTAINS: biểu diễn cấu trúc văn bản.
- REFERENCES: quan hệ dẫn chiếu.
- AMENDS: sửa đổi.
- SUPPLEMENTS: bổ sung.
- REPLACES: thay thế.
- REPEALS: bãi bỏ.
- DETAILS hoặc IMPLEMENTS: quan hệ quy định chi tiết hoặc thi hành khi có căn cứ.

Mỗi edge cần lưu source node, target node, relation_type, evidence text, nguồn chứng minh, thời điểm hoặc phiên bản liên quan nếu xác định được, phương thức phát hiện và trạng thái xác minh.

### 8.4. Xây dựng quan hệ

Ưu tiên:

1. Sử dụng quan hệ có cấu trúc trong dữ liệu chính thức nếu có.
2. Phân tích các dẫn chiếu rõ ràng trong văn bản, sau đó ánh xạ số hiệu và vị trí điều khoản đến định danh đã chuẩn hóa.
3. Với quan hệ phức tạp được phát hiện tự động, lưu trạng thái chưa xác minh cho đến khi có căn cứ.

Không được chuyển quan hệ do LLM suy đoán thành quan hệ pháp lý chính thức mà không có bước kiểm chứng.

### 8.5. Graph expansion

Khi người dùng gửi câu hỏi:

1. Thực hiện retrieval ban đầu bằng BM25, dense hoặc hybrid.
2. Xem các kết quả đầu tiên như seed nodes.
3. Mở rộng graph 1-hop trước.
4. Có thể cho phép 2-hop để thử nghiệm.
5. Cho phép chọn loại quan hệ được mở rộng.
6. Loại bỏ node trùng lặp.
7. Giới hạn số node bổ sung để tránh tăng context không kiểm soát.
8. Giữ lại lý do bổ sung từng node và đường đi từ seed đến node đó.
9. Xếp hạng lại hoặc chọn lọc kết quả bổ sung thay vì đưa tất cả quan hệ vào context một cách máy móc.

Phải có cấu hình KG Off và KG On. Khi bật/tắt KG, giữ nguyên seed retrieval, corpus, chunking và các thông số khác để đo được đóng góp riêng của graph expansion.

### 8.6. Kiểm soát hiệu lực

Nếu dữ liệu có thông tin phiên bản, thời gian hiệu lực hoặc quan hệ sửa đổi:

- Cho phép truyền ngày tham chiếu vào truy vấn.
- Áp dụng điều kiện thời gian cho tất cả các node được lấy vào context, không chỉ seed nodes.
- Không tự suy luận ngày kết thúc hiệu lực nếu nguồn không xác định.
- Nếu không đủ dữ liệu để xác định phiên bản áp dụng, phải hiển thị rõ trạng thái chưa xác định.

## 9. THIẾT KẾ BENCHMARK

Xây dựng tập benchmark gồm câu hỏi, căn cứ pháp lý chuẩn và metadata cần thiết.

Mỗi mẫu nên chứa:

- question_id.
- Question text.
- Relevant provision IDs.
- Document IDs liên quan.
- Các điều khoản liên quan được chấp nhận nếu câu hỏi cần nhiều căn cứ.
- Ngày tham chiếu nếu câu hỏi yêu cầu tra cứu lịch sử.
- Quan hệ kỳ vọng nếu câu hỏi dùng để đánh giá Knowledge Graph.
- Nguồn xác minh đáp án.

Bắt đầu bằng một bộ khoảng 50–100 câu hỏi được gán nhãn và kiểm tra thủ công. Nếu tìm được benchmark công khai phù hợp, có thể bổ sung để mở rộng đánh giá. Mỗi benchmark phải ghi rõ nguồn và phạm vi của bộ dữ liệu.

Tạo các nhóm câu hỏi:

1. Truy xuất trực tiếp một Điều hoặc Khoản.
2. Câu hỏi diễn đạt lại, ít trùng từ với nội dung pháp luật.
3. Câu hỏi cần theo dấu dẫn chiếu sang văn bản khác.
4. Câu hỏi cần nhiều căn cứ pháp lý cùng lúc.
5. Câu hỏi về điều khoản đã sửa đổi hoặc cần xác định hiệu lực tại một ngày cụ thể.

Nếu chưa có nhãn chuẩn thì vẫn cho phép chạy tìm kiếm, nhưng không được hiển thị chỉ số đánh giá như thể đã có ground truth.

## 10. CÁC CHỈ SỐ ĐÁNH GIÁ

Triển khai ít nhất:

- Recall@1, Recall@3, Recall@5, Recall@10.
- MRR@k.
- nDCG@k nếu bộ nhãn có mức độ liên quan.
- Provision Hit Rate: tỷ lệ truy vấn tìm được đúng điều khoản gốc.
- Graph Expansion Precision: tỷ lệ node bổ sung qua KG thực sự liên quan.
- Latency: thời gian truy xuất.
- Context size: số token hoặc số chunk được đưa vào context.

Nếu có benchmark temporal:

- Tỷ lệ chọn đúng phiên bản có hiệu lực tại ngày tham chiếu.
- Tỷ lệ chọn nhầm phiên bản đã hết hiệu lực hoặc chưa có hiệu lực.
- Tỷ lệ câu trả lời không xác định được phiên bản khi dữ liệu còn thiếu.

Metric phải được tính từ kết quả thực tế và ground truth. Không được hard-code các con số hoặc trình bày dữ liệu mẫu như kết quả benchmark thật.

Báo cáo cả kết quả tổng thể lẫn kết quả theo từng nhóm câu hỏi. Mục tiêu là biết khi nào mỗi kỹ thuật có ích, không chỉ tìm cấu hình có con số trung bình cao nhất.

## 11. GIAO DIỆN STREAMLIT

Xây dựng giao diện gọn, dễ trình bày trong buổi demo, gồm các khu vực sau.

### Tab 1 — Interactive Search

- Ô nhập câu hỏi.
- Bộ chọn retrieval mode: BM25, Dense, Hybrid.
- Bộ chọn chunking strategy.
- Công tắc Knowledge Graph On/Off.
- Bộ chọn độ sâu graph expansion.
- Bộ lọc loại quan hệ.
- Ô chọn ngày tham chiếu nếu có dữ liệu temporal.
- Nút chạy truy vấn.

Kết quả cần hiển thị:

- Các chunk được truy xuất.
- Số hiệu và tên văn bản.
- Điều/Khoản/Điểm.
- Trạng thái và mốc thời gian nếu có.
- Nguồn trích dẫn.
- Phương thức truy xuất.
- Lý do một tài liệu được bổ sung qua KG.

### Tab 2 — Chunking Lab

- Chọn một văn bản.
- Hiển thị cách văn bản được chia theo ba chiến lược.
- Hiển thị số lượng chunk và thống kê số token.
- Cho phép so sánh các kết quả retrieval trên cùng câu hỏi.

### Tab 3 — Knowledge Graph Explorer

- Trực quan hóa các node và edge.
- Hiển thị loại quan hệ.
- Có thể chọn một Điều/Khoản để xem các node liên quan.
- Hiển thị evidence và nguồn chứng minh mỗi quan hệ.
- Làm rõ node nào xuất phát từ semantic/keyword retrieval và node nào được bổ sung qua graph expansion.
- Nếu có phiên bản lịch sử, hiển thị chuỗi thay đổi liên quan.

### Tab 4 — Benchmark

- Chạy benchmark trên nhiều cấu hình.
- So sánh các metric theo cấu hình.
- Hiển thị biểu đồ và bảng kết quả.
- Lưu kết quả benchmark thành CSV hoặc JSON.
- Hiển thị thời gian build index, thời gian truy xuất và các thông số cấu hình liên quan.

Có thể bổ sung panel trả lời bằng LLM ở giai đoạn sau. Nếu triển khai, câu trả lời phải gắn với bằng chứng được truy xuất và không đưa ra trích dẫn không tồn tại.

## 12. MA TRẬN THÍ NGHIỆM

Triển khai ma trận thử nghiệm ban đầu:

- 3 chiến lược chunking:
  - Article-level.
  - Fixed-size.
  - Structure-aware.
- 3 retrieval modes:
  - BM25.
  - Dense.
  - Hybrid.
- 2 trạng thái Knowledge Graph:
  - KG Off.
  - KG On.

Tổng cộng 18 cấu hình cơ bản, sử dụng chung tập câu hỏi và ground truth.

Sau khi có kết quả ban đầu, có thể bổ sung thí nghiệm về embedding model, graph depth, relation types và temporal retrieval.

Để bảo đảm công bằng:

- Chạy trên cùng corpus.
- Dùng cùng bộ câu hỏi.
- So sánh với cùng định nghĩa ground truth.
- Cố định top-k và ngân sách context cuối cùng.
- Ghi lại model version, tham số và thời gian build index.
- Không chọn riêng các câu hỏi mà cấu hình đề xuất làm tốt hơn.
- Nếu có yếu tố không thể cố định, ghi rõ trong báo cáo.

## 13. KIẾN TRÚC MÃ NGUỒN ĐỀ XUẤT

Có thể tổ chức thư mục theo hướng sau, nhưng hãy điều chỉnh dựa trên workspace thực tế thay vì tạo cấu trúc mới một cách máy móc.

- `app/`: giao diện Streamlit.
- `ingestion/`: đọc, chuẩn hóa và xác thực dữ liệu.
- `chunking/`: các chiến lược chunking.
- `embedding/`: model adapter, batching, caching và embedding generation.
- `retrieval/`: BM25, dense retrieval, hybrid fusion.
- `knowledge_graph/`: xây dựng node/edge, phân tích dẫn chiếu và graph expansion.
- `temporal/`: phiên bản và lọc theo ngày tham chiếu.
- `evaluation/`: ground truth, metric và benchmark runner.
- `storage/`: truy cập database và index.
- `data/`: dữ liệu mẫu hoặc đường dẫn dữ liệu được cấu hình.
- `artifacts/`: index, embedding cache và kết quả benchmark.
- `tests/`: unit tests và integration tests.
- `config/`: cấu hình model, chunking, retrieval và graph.
- `README.md`: hướng dẫn cài đặt, build index, chạy demo và benchmark.

Tách phần UI khỏi logic xử lý để dễ kiểm thử. Hạn chế viết toàn bộ pipeline trong một file.

## 14. TEST VÀ KIỂM SOÁT CHẤT LƯỢNG

Tạo các test quan trọng:

1. Parser giữ đúng mối quan hệ Điều/Khoản/Điểm.
2. Chunking không âm thầm làm mất văn bản.
3. Chunk có thể truy ngược về đúng điều khoản gốc.
4. Embedding index ánh xạ đúng với ID tài liệu.
5. BM25, dense và hybrid trả về đúng định dạng.
6. KG expansion tìm đúng node liên quan và không lặp node.
7. Quan hệ chưa xác minh không được trình bày như quan hệ chính thức.
8. Ngày tham chiếu được áp dụng cho cả seed và node mở rộng.
9. Tài liệu thiếu metadata được xử lý có kiểm soát.
10. Benchmark tính metric đúng trên các bộ test nhỏ đã biết kết quả.
11. Chạy lại ingestion không tạo bản ghi trùng ngoài ý muốn.
12. Kết quả có thể truy xuất về nguồn gốc văn bản.

Không chỉ chạy ứng dụng bằng tay. Cần có test tự động cho logic chính và một vài integration test cho pipeline từ dữ liệu tới retrieval.

## 15. TIÊU CHÍ HOÀN THÀNH

Demo được coi là hoàn thành khi:

- Có thể cài đặt và chạy local theo README.
- Có thể ingest một tập dữ liệu pháp luật thực tế hoặc bộ dữ liệu mẫu được ghi nhãn rõ ràng.
- Có ít nhất ba chiến lược chunking.
- Có BM25, dense và hybrid retrieval.
- Có thể bật/tắt Knowledge Graph trên cùng seed retrieval.
- Có thể xem các quan hệ được duyệt và nguồn chứng minh.
- Có thể chạy benchmark và so sánh metric của các cấu hình.
- Có thể xuất kết quả benchmark.
- Mọi chunk truy xuất được đều liên kết với văn bản và vị trí pháp lý gốc.
- Có test tự động cho các chức năng quan trọng.
- Không có metric giả, nguồn pháp luật giả hoặc quan hệ pháp lý chưa xác minh được trình bày như dữ kiện chính thức.
- Ghi rõ các hạn chế của dữ liệu, benchmark và mô hình.

## 16. THỨ TỰ THỰC HIỆN

Triển khai theo các giai đoạn sau:

### Phase 0 — Khảo sát

Kiểm tra workspace, dữ liệu và môi trường hiện có. Viết ngắn gọn kế hoạch kỹ thuật dựa trên trạng thái thực tế, sau đó bắt đầu triển khai.

### Phase 1 — Data ingestion và baseline

Hoàn thiện ingestion, metadata schema, article-level chunking, BM25 retrieval và pipeline đánh giá cơ bản. Bảo đảm có thể chạy một truy vấn từ đầu đến cuối.

### Phase 2 — Chunking và embedding

Triển khai các chiến lược chunking, persistent embedding index, dense retrieval và hybrid retrieval. Xây dựng các thí nghiệm so sánh.

### Phase 3 — Knowledge Graph

Xây dựng graph schema, phát hiện các dẫn chiếu có căn cứ, lưu evidence, graph expansion và chế độ KG Off/On.

### Phase 4 — Benchmark

Hoàn thiện bộ câu hỏi có ground truth, metric, ma trận thí nghiệm, lưu kết quả và tạo báo cáo so sánh.

### Phase 5 — Streamlit UI

Hoàn thiện bốn tab chính: Interactive Search, Chunking Lab, Knowledge Graph Explorer và Benchmark.

### Phase 6 — Kiểm thử và tài liệu

Chạy test, sửa lỗi, kiểm tra tính đồng bộ của dữ liệu và index, viết README hướng dẫn sử dụng. Nêu rõ những chức năng đã hoàn thành và những chức năng chưa thể triển khai do thiếu dữ liệu hoặc phụ thuộc.

Ưu tiên một MVP chạy được từ đầu tới cuối trước. Không dành quá nhiều thời gian thiết kế UI hoặc xây graph phức tạp khi baseline retrieval và benchmark chưa hoạt động.

## 17. YÊU CẦU LÀM VIỆC CHO CODING AGENT

- Hãy trực tiếp kiểm tra workspace và triển khai mã nguồn; không chỉ trả về một bản thiết kế.
- Sử dụng lại các thành phần đang hoạt động nếu phù hợp.
- Không ghi đè hoặc xóa dữ liệu hiện có nếu chưa kiểm tra tác động.
- Không yêu cầu API key khi có thể dùng giải pháp local cho MVP.
- Không đưa secret vào mã nguồn.
- Không hard-code kết quả benchmark.
- Không tuyên bố demo có hiệu quả cao hơn baseline nếu chưa có kết quả đo thực tế.
- Không tự động cài đặt các dependency nặng hoặc tải model lớn mà không ghi lại yêu cầu tài nguyên.
- Với mỗi giai đoạn, chạy test hoặc kiểm tra chức năng trước khi tiếp tục.
- Khi gặp dữ liệu thiếu hoặc không xác định được quan hệ, hãy xử lý theo hướng an toàn, ghi log và mô tả hạn chế thay vì tạo dữ liệu giả.
- Ưu tiên mã nguồn dễ hiểu, module độc lập, cấu hình rõ ràng và có thể mở rộng.

Khi hoàn thành, hãy cung cấp:
1. Tóm tắt kiến trúc đã triển khai.
2. Hướng dẫn cài đặt và chạy demo.
3. Hướng dẫn ingest dữ liệu và build index.
4. Hướng dẫn chạy benchmark.
5. Kết quả test thực tế.
6. Kết quả benchmark thực tế nếu đã có bộ dữ liệu đủ điều kiện.
7. Những hạn chế còn tồn tại và bước tiếp theo nên làm.