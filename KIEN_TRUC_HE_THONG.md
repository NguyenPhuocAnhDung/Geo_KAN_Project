# KIẾN TRÚC HỆ THỐNG VÀ LOGIC MẠNG NƠ-RON (DRIFT-TKAN)

Tài liệu này đi sâu vào chi tiết kỹ thuật cơ sở của toàn bộ luồng vận hành hệ thống (System Pipeline) và cách thức cấu trúc mạng nơ-ron (Network Architecture) được thiết kế để giải quyết bài toán An ninh mạng trong môi trường dữ liệu biến thiên (Concept Drift).

---

## PHẦN 1: LOGIC KIẾN TRÚC TỔNG THỂ CỦA HỆ THỐNG (SYSTEM ARCHITECTURE PIPELINE)

Hệ thống Drift-TKAN không chỉ là một mô hình học sâu đơn lẻ, mà là một **Đường ống học liên tục (Continual Learning Pipeline)** vận hành theo thời gian thực. Kiến trúc tổng thể được chia thành 4 giai đoạn cốt lõi:

### 1. Giai đoạn Tiền xử lý và Đóng gói (Data Stream Processing)
Do đặc thù của mạng máy tính, các gói tin không tồn tại độc lập mà có tính liên tục theo thời gian.
- **Lọc đặc trưng (Feature Selection):** Lựa chọn 11 đặc trưng toàn cục (Global Features) tốt nhất đại diện cho giao thức TCP/UDP, DNS, ARP.
- **Trượt Cửa sổ Thời gian (Time-series Windowing):** Các bản ghi đơn lẻ được gộp lại thành các chuỗi thời gian (Sequence) với độ dài $L = 10$. Điều này giúp mô hình nhận diện được "nhịp độ" của một cuộc tấn công (ví dụ: Tấn công DDoS luôn có tần suất gói tin ồ ạt trong một khung thời gian).
- **Phân mảnh luồng (Chunking):** Luồng dữ liệu vô tận được chia cắt thành các Khối (Chunks). Tại đây, mỗi Chunk được cài đặt cứng là $10,000$ mẫu. Việc xử lý theo Chunk mô phỏng hoàn hảo môi trường Stream thực tế.

### 2. Giai đoạn Khởi tạo (Phase 1 - Base Knowledge Initialization)
Hệ thống cần một "trí nhớ gốc" để biết thế nào là mạng bình thường và thế nào là các dạng tấn công cơ bản.
- Mô hình được huấn luyện ngoại tuyến (Offline Training) trên một tập dữ liệu ban đầu (Phase 1).
- Trọng số và cấu trúc nội tại của mô hình tại thời điểm này được đóng băng và lưu trữ làm **Đường cơ sở (Baseline Representation $\mathbf{C}_{base}$)**.

### 3. Giai đoạn Giám sát Hộp trắng (Streaming & White-box Monitoring)
Đây là giai đoạn dòng chảy mạng thực tế đổ về liên tục (Phase 2 & Phase 3).
Tại mỗi Chunk mới (Chunk $t$):
- **Phản ứng nhanh (Test-then-Train):** Mô hình đưa ra dự đoán (Inference) để lấy F1-score thực tế. Ngay sau đó, mô hình tự tinh chỉnh nhẹ (Fine-tuning với 1 Epoch) trên chính Chunk $t$ này.
- **Trích xuất Cảm biến (Sensor Extraction):** Thuật toán sẽ chui vào bên trong lớp KAN cuối cùng, trích xuất ma trận hệ số Chebyshev hiện tại $\mathbf{C}_t$.
- **Đo lường Khoảng cách (Shift Score):** Tính toán khoảng cách cấu trúc bằng Cosine Similarity giữa $\mathbf{C}_{base}$ và $\mathbf{C}_t$.

### 4. Giai đoạn Phân xử và Cập nhật (Thresholding & Adaptation)
- Điểm Shift Score được đưa qua bộ lọc **Trung bình trượt mũ (EMA)** với ngưỡng **3-Sigma**.
- **Nếu KHÔNG có Drift:** Hệ thống coi sự biến thiên chỉ là nhiễu tự nhiên, đường cơ sở (Baseline) được cập nhật trượt từ từ (với $\alpha = 0.99$) để thích nghi với sự thay đổi nhỏ lẻ.
- **Nếu CÓ Drift (Mã độc mới xuất hiện):**
  - Hệ thống phát ra tín hiệu cảnh báo (Drift Detected).
  - Kích hoạt **Replay Buffer**: Hệ thống lấy ngẫu nhiên 500 mẫu cũ từ bộ nhớ trộn với dữ liệu dị thường mới.
  - Tái huấn luyện (Retrain) để mô hình cập nhật kiến thức về mã độc mới mà không bị mắc hội chứng "Quên thảm họa" (Catastrophic Forgetting) những mã độc cũ.

---

## PHẦN 2: LOGIC KIẾN TRÚC MẠNG NƠ-RON (NETWORK ARCHITECTURE LOGIC)

Kiến trúc bên trong của Drift-TKAN (Class `HierarchicalDriftTKAN`) là một bản giao hưởng giữa xử lý chuỗi thời gian (BiLSTM + Attention) và không gian hàm học được (Chebyshev KAN). 

Cấu trúc luồng truyền xuôi (Forward Pass) đi qua các khối sau:

### Khối 1: Tầng trích xuất Ngữ cảnh Không gian - Thời gian (BiLSTM)
- **Đầu vào (Input Tensor):** $\mathbf{X} \in \mathbb{R}^{B \times L \times F}$ (Với $B$: Batch size, $L$: Độ dài chuỗi = 10, $F$: Số lượng đặc trưng = 11).
- **Thực thi:** Bidirectional LSTM duyệt qua chuỗi thời gian theo 2 chiều (từ quá khứ đến hiện tại và ngược lại).
- **Ý nghĩa Logic:** Mã độc thường giấu hành vi của chúng trong một chuỗi các thao tác (như gửi gói tin mồi, mở cổng kết nối, rồi mới bơm dữ liệu). BiLSTM giúp kết nối các mắt xích hành vi này lại thành một vector ẩn đại diện (Hidden State) $\mathbf{H} \in \mathbb{R}^{B \times L \times (2 \times \text{Hidden\_Size})}$.

### Khối 2: Tầng Chú ý Nhịp độ (Temporal Self-Attention)
- **Cơ chế (Mechanism):** Không phải gói tin nào trong chuỗi 10 gói tin cũng chứa dấu hiệu độc hại. Cơ chế Self-Attention sẽ gán trọng số (Weights $\alpha_i$) cho từng gói tin dựa trên mức độ bất thường của nó so với toàn bộ chuỗi.
- **Đầu ra (Output Tensor):** Tạo ra một Vector Ngữ cảnh duy nhất (Context Vector) $\mathbf{c} \in \mathbb{R}^{B \times (2 \times \text{Hidden\_Size})}$, chứa đựng toàn bộ tinh hoa của chuỗi tấn công.

### Khối 3: Tầng Giảm chiều và Làm mượt (MLP Bottleneck)
- **Thực thi:** Đưa Vector Ngữ cảnh $\mathbf{c}$ đi qua một mạng truyền thẳng (Linear + LayerNorm + SiLU/GELU).
- **Ý nghĩa Logic:** Đóng vai trò như một phễu lọc nhiễu, ép các đặc trưng thô thành một biểu diễn cô đọng (Condensed Representation) ổn định hơn trước khi đưa vào KAN. Điều này giúp hệ số KAN ở tầng sau không bị nhiễu loạn (Over-sensitive) bởi các biến động nhỏ giọt của mạng máy tính.

### Khối 4: Tầng Phân loại Hộp trắng (Chebyshev KAN Layer)
Đây là "trái tim" của thuật toán. Thay vì sử dụng lớp Linear (nhân ma trận trọng số $W \cdot x + b$) như các mô hình IDS thông thường, lớp KAN sử dụng cấu trúc Hàm toán học.

- **Cấu trúc:** Gồm $N_{in}$ nơ-ron đầu vào và $N_{out}$ nơ-ron đầu ra (15 nhãn phân loại). Bất kỳ nơ-ron đầu vào $i$ nào kết nối với nơ-ron đầu ra $j$ đều thông qua một hàm đa thức Chebyshev $\phi_{i,j}(x)$.
- **Logic Tính toán (Forward):**
  1. Dữ liệu $x$ được ánh xạ qua các đa thức Chebyshev $T_k(x)$ với bậc $q = 3$.
  2. Các hệ số $c_{i,j,k}$ được nhân với $T_k(x)$ để tính ra giá trị hàm.
  3. Tổng hợp lại để đưa ra Logits dự đoán (Phân loại ra 1 trong 15 loại Tấn công/Bình thường).
- **Logic Bắt Drift (Structural Diagnosis):**
  Lớp KAN này không chỉ dự đoán. Khi mô hình học một khái niệm mới (Ví dụ: Hành vi DDoS ngày hôm nay khác với ngày hôm qua), để biểu diễn được sự khác biệt này, mạng nơ-ron **bắt buộc phải bẻ cong các hàm Chebyshev**. Do đó, các hệ số (Coefficients) của KAN bị uốn nắn theo. 
  Hệ thống sẽ đo lường mức độ uốn nắn này (thông qua hàm `compute_hierarchical_drift`) để làm chỉ báo (Indicator) báo động Drift.

### Tóm tắt Luồng Dữ liệu (Forward Flow Summary):
1. **Raw Network Flow** $\xrightarrow{\text{Windowing}}$ **Sequence (10, 11)**
2. **Sequence** $\xrightarrow{\text{BiLSTM}}$ **Hidden Temporal States**
3. **Hidden States** $\xrightarrow{\text{Attention}}$ **Context Vector**
4. **Context Vector** $\xrightarrow{\text{MLP Bottleneck}}$ **Condensed Features**
5. **Condensed Features** $\xrightarrow{\text{KAN Layer}}$ **15-Class Logits** & **Chebyshev Coefficients Matrix** (Dùng cho đo lường Drift).
