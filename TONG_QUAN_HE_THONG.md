# TỔNG QUAN HỆ THỐNG: DRIFT-TKAN CONTINUAL LEARNING

Tài liệu này tổng hợp thiết kế và logic cốt lõi của Framework **Drift-TKAN** - một giải pháp phát hiện trôi dạt khái niệm (Concept Drift) theo cơ chế hộp trắng (White-box) dành cho lĩnh vực An toàn thông tin mạng (Cybersecurity). Đây là bộ khung học thuật vững chắc hướng tới các ấn phẩm khoa học Q1/Top-tier.

## 1. Kiến trúc Cốt lõi (Core AI Architecture)

Mô hình học sâu kết hợp giữa Học biểu diễn thời gian (Temporal Learning) và Kolmogorov-Arnold Networks (KAN):

- **BiLSTM (Bidirectional LSTM)**: Khai phá ngữ cảnh hai chiều của luồng mạng, trích xuất đặc trưng chuỗi thời gian (Temporal Features).
- **Self-Attention Mechanism**: Gắn trọng số nhịp độ thời gian để phân biệt rõ ràng giữa các cuộc tấn công DDoS/DoS (tần suất cao) và Benign (hoạt động bình thường).
- **MLP Bottleneck**: Ép dữ liệu và làm mượt không gian đặc trưng.
- **Chebyshev KAN Layer**: 
  - Lớp phân loại phi tuyến tính thay thế cho Linear thông thường. 
  - **Đóng góp cốt lõi (Novelty)**: KAN Layer không chỉ phân loại, mà sự thay đổi trong hệ số Chebyshev của nó đóng vai trò là "Cảm biến" đo lường mức độ méo mó của phân phối dữ liệu (Concept Drift).

## 2. Logic Phát hiện Trôi dạt Hộp trắng (White-box Drift Detection)

Thay vì chờ độ chính xác (Accuracy) giảm sút mới bắt đầu phản ứng (như các thuật toán hộp đen ADWIN, Page-Hinkley), Drift-TKAN giám sát sự biến thiên của cấu trúc nội tại mô hình.

1. **Trọng số Tham chiếu (Baseline/EMA Coeffs)**: Hệ số KAN được duy trì một bản sao sử dụng Đường trung bình động hàm mũ (Exponential Moving Average - EMA) với $\alpha = 0.99$. Đây là "ký ức" về trạng thái ổn định của dữ liệu cũ.
2. **Khoảng cách Phân phối (Shift Score)**: Với mỗi Chunk dữ liệu mới đến, mô hình được "thích nghi nhẹ" (Light Adaptation) bằng cách chỉ fine-tune lớp KAN. Sau đó, tính **Cosine Distance** giữa hệ số KAN mới và hệ số EMA cũ. Khoảng cách này chính là Shift Score.
3. **Ngưỡng Động 3-Sigma (Dynamic Statistical Threshold)**:
   - Ngưỡng kích hoạt $\tau$ không được chọn thủ công (hard-code) để tránh việc cố tình cherry-picking dữ liệu.
   - $\tau = \mu_{shift} + 3\sigma_{shift}$ (Trung bình cộng 3 lần Độ lệch chuẩn của độ trôi dạt ở Phase 1 - Giai đoạn Normal). Bất kỳ Shift Score nào vượt $\tau$ (ví dụ 0.025) đều được khẳng định là một sự dịch chuyển phân phối có ý nghĩa thống kê (Statistical Process Control).

## 3. Chiến lược Đánh giá và Huấn luyện Liên tục (Continual Learning Pipeline)

Thiết kế thực nghiệm tuân thủ nghiêm ngặt chuẩn **Prequential Evaluation (Interleaved Test-Then-Train)** để mô phỏng chính xác kịch bản Online trong thực tế.

- **Bước 1: Prequential Test**: Dự đoán và ghi nhận F1-Score trên Chunk mới ngay lập tức (khi mô hình chưa hề biết dữ liệu này). Đảm bảo tính công bằng.
- **Bước 2: Light Adaptation**: Đóng băng BiLSTM, chỉ mở KAN và Attention. Train nhanh 1 Epoch (lr=1e-4).
- **Bước 3: Drift Measurement**: Tính Cosine Shift Score. 
- **Bước 4: Online Adaptation (Fine-tuning)**:
  - Nếu `Shift Score < \tau`: Cập nhật EMA và đi tiếp.
  - Nếu `Shift Score > \tau`: Kích hoạt cảnh báo **🔴 DRIFT DETECTED**. Tiến hành Mix (pha trộn) dữ liệu Chunk mới với **Class-Balanced Replay Buffer**.
- **Bước 5: Replay Buffer Update**: Trích xuất ngẫu nhiên dữ liệu mới (theo Reservoir Sampling) nạp vào Buffer để chống lại hiện tượng Quên thảm họa (Catastrophic Forgetting).

## 4. Các Biện pháp Ổn định Kỹ thuật (Technical Stability)

Để giải bài toán hàng trăm triệu bản ghi dữ liệu mạng (Data Streams) mà không bị "chết" mô hình, hệ thống áp dụng:
1. **Numerically Stable Focal Loss**: Sử dụng `F.log_softmax` và `F.nll_loss` thay cho `F.cross_entropy` kết hợp số mũ, giúp dập tắt hoàn toàn rủi ro tràn số (NaN Loss) do các nhãn dữ liệu mất cân bằng cực đoan (Long-tail distribution).
2. **Xử lý Nhiễu (NaN/Inf Padding)**: Làm sạch luồng numpy (`np.nan_to_num`) ngay từ lớp Dataloader trước khi đi vào tính toán đạo hàm.
3. **Freeze/Unfreeze linh hoạt**: Giải phóng tài nguyên VRAM/RAM và giới hạn vùng biến đổi gradient khi có Drift.

---
*Tài liệu này được định hướng làm đề cương trực tiếp cho phần Methodology và Evaluation của bài báo Q1.*

## 5. Hệ thống Công thức Toán học (Mathematical Formulation)

### A. Tầng BiLSTM & Self-Attention
Cho một chuỗi đặc trưng đầu vào $X = [x_1, x_2, ..., x_T]$ (với $T=10$ là `SEQ_LENGTH`):
1. **BiLSTM** trích xuất đặc trưng hai chiều:
   $$ h_t = \text{BiLSTM}(x_t, h_{t-1}) $$
2. **Self-Attention** gán trọng số nhịp độ (Temporal Importance):
   $$ e_t = W_a h_t + b_a $$
   $$ \alpha_t = \frac{\exp(e_t)}{\sum_{k=1}^{T} \exp(e_k)} $$
   $$ C = \sum_{t=1}^{T} \alpha_t h_t $$
   Trong đó $C$ là Context Vector chứa đặc trưng cô đọng của chuỗi.

### B. Tầng Chebyshev KAN
Với đầu vào $z$ từ lớp MLP ($z = \text{MLP}(C)$), KAN Layer dự đoán nhãn thông qua đa thức Chebyshev loại 1:
1. Khởi tạo cơ sở Chebyshev: $T_0(z) = 1$, $T_1(z) = z$
   $$ T_n(z) = 2zT_{n-1}(z) - T_{n-2}(z) $$
2. Kết xuất logits:
   $$ \hat{y}_j = \sum_{i=1}^{D} \sum_{n=0}^{Degree} w_{jin} T_n(z_i) $$
   Với $w$ là ma trận trọng số (Chebyshev coefficients) của KAN.

### C. Đo lường Concept Drift
Dưới các giả định nhẹ (mild assumptions), sự thay đổi của các hệ số Chebyshev có tương quan chặt chẽ với sự dịch chuyển của hàm ranh giới quyết định (Decision Boundary Shift). Do đó, ta đo lường Drift thông qua hệ số KAN thay vì đo trên prediction:
1. **EMA Update (Exponential Moving Average)**:
   $$ W_{base}^{(t)} = \alpha W_{base}^{(t-1)} + (1-\alpha) W_{curr}^{(t)} $$
   *(Sử dụng $\alpha = 0.99$ làm hệ số suy giảm để tạo đường tham chiếu cơ sở).*
2. **Shift Score (Cosine Distance)**:
   $$ \text{Shift Score} = 1 - \frac{W_{curr} \cdot W_{base}}{\|W_{curr}\| \|W_{base}\|} $$
3. **Ngưỡng Động (Dynamic Threshold - 3-Sigma Rule)**:
   Tại Phase 1 (dữ liệu ổn định), tính trung bình $\mu_{shift}$ và độ lệch chuẩn $\sigma_{shift}$:
   $$ \tau = \mu_{shift} + 3\sigma_{shift} $$
   *(Khi $\text{Shift Score} > \tau$, hệ thống kích hoạt cơ chế Online Adaptation).*

### D. Numerically Stable Focal Loss
Để chống lại sự mất cân bằng dữ liệu cực đoan và tránh tràn số (NaN Loss):
$$ \mathcal{L}_{focal} = - \frac{1}{N} \sum_{i=1}^{N} \alpha_{y_i} (1 - P(y_i|x_i))^\gamma \log P(y_i|x_i) $$
Trong đó $\log P(y_i|x_i)$ được tính toán an toàn thông qua hàm `F.log_softmax()`.

---

## 6. Bộ Dữ liệu Thực nghiệm (Datasets Pipeline)

Mô hình được thử nghiệm bằng luồng dữ liệu mạng liên tục vắt ngang qua nhiều năm, mô phỏng sự tiến hóa thực tế của các loại mã độc và tấn công mạng:

1. **Phase 1 (Train Baseline - Khởi tạo Stable State):**
   - **CIC-IDS-2017**: Làm nền tảng học các dạng tấn công cơ bản (DDoS, DoS, Web Attack, Infiltration...). Cung cấp dữ liệu làm khuôn cho Scaler gồm 114 đặc trưng và 9 nhãn tổng hợp.
2. **Phase 2 (Drift Test 1 - IoT & Darknet):**
   - **CIC-IDS-2018 (CSE-CIC-IDS2018)**: Sự biến hóa của các luồng botnet hiện đại.
   - **CICDarknet2020**: Tấn công mã hóa qua nền tảng ẩn danh.
   - **CICIoT2023**: Môi trường Internet vạn vật (IoT) với kiến trúc gói tin khác biệt.
3. **Phase 3 (Domain Shift - Tấn công Xe điện EVSE 2024):**
   - **CICEVSE2024**: Dữ liệu từ trạm sạc xe điện (EVSE). Đây là môi trường hoàn toàn mới (Domain Shift), kiểm tra tính khắc nghiệt nhất của thuật toán khi đối mặt với lượng lớn gói tin chưa từng thấy.

**Cấu trúc Đầu vào thống nhất (Union Schema)**: Toàn bộ các bộ dataset trên được map về cùng chung một tập hợp $114$ Features và $9$ Nhãn chuẩn hóa: `[Benign, Botnet, Brute_Force, DDoS, DoS, Infiltration, Other_Attack, PortScan, Web_Attack]`.
*Lưu ý về Dataset Leakage: Các tính năng bị thiếu (missing features) khi chuyển miền (vd: từ 2017 sang EVSE) được xử lý bằng global mean imputation / zero-padding dựa hoàn toàn trên global_scaler của Phase 1, đảm bảo không có information leakage.*

---

## 7. Thiết kế Thực nghiệm TIFS Q1 (Cập nhật 2026)

Để thuyết phục các tạp chí hàng đầu (IEEE TIFS, TDSC), dự án đã chuyển trọng tâm từ "Architecture Novelty" sang việc **Thiết kế thực nghiệm để kiểm định các giả thuyết khoa học**. Dưới đây là lộ trình 6 bước (6 Scripts) nhằm cung cấp Bằng chứng thép (Evidence-based):

### Giai đoạn 1: Bảo vệ 3 Giả thuyết Cốt lõi
- **H1 (Mối Tương Quan - `experiment_correlation.py`)**: Kiểm định giả thuyết Shift Score (Chebyshev Coefficients) phản ánh sự trôi dạt khái niệm và là một tín hiệu Early Warning.
  - *Phương pháp*: Theo dõi Shift Score, F1, Loss, ECE, Entropy. Tính toán Pearson (r), Spearman ($\rho$) và Cross-Correlation (Lag phase).
- **H2 (Phát hiện Sớm - `experiment_latency_baselines.py`)**: Kiểm định giả thuyết Drift-TKAN cho phép thích nghi nhanh hơn các hệ thống cảnh báo hộp đen.
  - *Phương pháp*: So sánh **Time to Actionable Adaptation** giữa các Detector (ADWIN, PH, DDM dùng error stream vs Drift-TKAN dùng phân phối). Đo lường `Detection Delay`, `Max F1 Drop`, `Recovery Chunks`. Sử dụng Effect Size (Cohen's d) và Wilcoxon Signed-Rank Test.
- **H3 (Tính Thiết Yếu - `experiment_ablation.py`)**: Kiểm định giả thuyết lợi ích đến từ chính kiến trúc được đề xuất.
  - *Phương pháp*: Strict Variable Control. So sánh Architecture (BiLSTM+KAN vs BiLSTM+MLP) và Drift Sensitivity ($\alpha, k\sigma$, Replay Size).

### Giai đoạn 2: Phân tích Chuyên sâu (Bonus Analysis)
- **H4 (Tính Minh Bạch - `experiment_explainability.py`)**: Bóc tách hộp đen bằng cách trực quan hóa Top-k hệ số Chebyshev thay đổi mạnh nhất (Heatmap, PCA) để chỉ rõ Layer/Neuron nào báo động Drift.
- **H5 (Chi Phí Thuật Toán - `experiment_complexity.py`)**: Tách bạch minh bạch giữa *Model Complexity* (Parameters, FLOPs, Inference time) và *Detector Overhead* (CPU time tính Shift Score so với thời gian duy trì Window của ADWIN).
- **H6 (Sự Bền Bỉ - `experiment_failure_generalization.py`)**: Bơm nhiễu (Noise) để test False Alarms (báo cáo chỉ số Mean Time Between False Alarms - MTBFA). Chạy Generalization bằng cách đảo thứ tự các Dataset (ví dụ: 2017 -> IoT -> EVSE -> Darknet) để xem mô hình có phụ thuộc vào thứ tự luồng dữ liệu hay không.

---

## 8. Nhật ký Cập nhật Kỹ thuật (Technical Changelog)

**Bản vá (Hotfix) Core Pipeline:**
1. **Mix Buffer Dimension Mismatch (Fixed)**: 
   - Lỗi `Sizes of tensors must match` ở Phase 2 (Chunk 785) đã được giải quyết. Nguyên nhân do hàm `.unfold()` kết hợp cắt chỉ số gây lệch Batch Size của luồng Tensors.
   - Đã thay thế bằng hàm `next(iter(chunk_loader))` an toàn hơn để bốc Batch ngẫu nhiên đồng bộ trực tiếp, kết hợp bắt lỗi `StopIteration`.
2. **Context Spatiotemporal Padding trong Replay (Fixed)**: 
   - Lỗi BiLSTM nhận dữ liệu "chuỗi tĩnh" ảo (timestep cuối repeat 10 lần) khi lấy từ Replay Buffer.
   - Đã được cập nhật để giữ nguyên trạng thái tensor 3 chiều `[Batch, SEQ_LENGTH, Features]`. Replay Buffer giờ đây học được 100% thuộc tính dịch chuyển thời gian của gói tin mạng. Mọi sai lệch Concept (Conceptual Flaw) đã bị loại bỏ.
