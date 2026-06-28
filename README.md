# Drift-TKAN: A White-box Concept Drift Detector for Cybersecurity

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Drift-TKAN** là một framework phát hiện trôi dạt khái niệm (Concept Drift Detection) hoàn toàn mới, được thiết kế đặc biệt cho các luồng dữ liệu mạng tốc độ cao (High-speed Network Traffic) trong lĩnh vực Cybersecurity. Bằng việc kết hợp Mạng nơ-ron học biểu diễn thời gian (Temporal Learning) và Mạng Kolmogorov-Arnold (KAN), Drift-TKAN mang lại khả năng chẩn đoán xâm nhập vượt trội mà không cần dùng đến các bộ phát hiện "Hộp đen" (Black-box) truyền thống.

---

## 🌟 Điểm nổi bật (Key Contributions)
1. **Phát hiện Drift Hộp trắng (White-box Detection):** Sử dụng sự biến thiên của hệ số đa thức Chebyshev trong KAN Layer để đo lường độ phân kỳ phân phối trực tiếp từ không gian trọng số (Weight Space).
2. **Siêu tốc độ (Zero-Overhead):** Thay vì duy trì một Cửa sổ trượt (Sliding Window) chứa Error Rate cồng kềnh như ADWIN hay DDM, hệ thống so sánh các ma trận Chebyshev trực tiếp trên GPU. Tốc độ kiểm tra 10,000 mẫu chỉ tốn **0.007ms** (nhanh hơn ADWIN ~10,000 lần).
3. **Chống "Quên kiến thức" (Catastrophic Forgetting):** Tích hợp Replay Buffer ngẫu nhiên khi xảy ra Drift, giúp mô hình phục hồi F1 tức thì mà không đánh mất kiến thức về các kiểu tấn công mạng trong quá khứ.
4. **Khả năng diễn giải (Explainability):** Dễ dàng trích xuất và visualize các hệ số nơ-ron nào đang nhạy cảm nhất với mã độc thông qua PCA và Heatmap.

---

## 📊 Tập dữ liệu (Dataset)
Sử dụng bộ dữ liệu mạng vạn vật IoT thực tế khổng lồ **Edge-IIoTset**:
- **Quy mô:** Hơn **11.5 triệu bản ghi** mạng (chia thành 786 chunks liên tục).
- **Đặc trưng (Features):** 11 đặc trưng cốt lõi (Global Features) được chọn lọc từ 61 đặc trưng gốc.
- **Nhãn (Classes):** 15 nhãn phân loại bao gồm Benign và 14 họ mã độc nguy hiểm (DDoS, DoS, Botnet, Web Attack, Port Scan...).

---

## 🧠 Kiến trúc Mô hình & Thuật toán
1. **BiLSTM:** Trích xuất dòng thời gian $\mathbf{h}_t = \text{BiLSTM}(\mathbf{x}_t)$.
2. **Self-Attention:** Tập trung vào các gói tin dị thường theo nhịp độ $\mathbf{c} = \text{Attention}(\mathbf{h})$.
3. **MLP Bottleneck:** Nén và làm mượt vector.
4. **Chebyshev KAN:** Lớp phi tuyến sử dụng đa thức $T_i(x)$ thay thế cho Activation function thông thường. Ma trận cấu trúc Chebyshev $\mathbf{C}_t$ được dùng để chẩn đoán.
   - **Shift Score:** $Shift = 1 - \text{Cosine}(\mathbf{C}_{base}, \mathbf{C}_t)$
   - **Động học 3-Sigma:** Kích hoạt Replay Buffer nếu $Shift > \mu_t + 3\sigma_t$.

---

## 📈 Kết quả Thực nghiệm Toàn diện (H1 - H6)

Hệ thống đã trải qua 6 thử thách thực nghiệm khắt khe nhất để chứng minh độ vững chãi cho bài báo Q1:

| Kịch bản Thực nghiệm | Kết quả Đạt được | Ý nghĩa Khoa học |
| :--- | :--- | :--- |
| **H1: Phân tích Tương quan** | Hệ số Spearman $\rho = -0.74$, Optimal Lag = 3 chunks. | Shift Score của KAN tỷ lệ nghịch cực mạnh với độ tụt giảm F1. Đóng vai trò làm chỉ báo trễ trung thực và đáng tin cậy. |
| **H2: So sánh Tốc độ Thích nghi** | Phục hồi F1 gần như lập tức (0 Recovery Chunks) tại các điểm gãy. | Vượt trội hoàn toàn so với các phương pháp Baseline như DDM và ADWIN. |
| **H3: Tối ưu Siêu tham số (Ablation)**| EMA $\alpha = 0.99$ và Ngưỡng 3-Sigma đạt F1 cao nhất ($0.8728$). | Cập nhật đường cơ sở quá nhanh ($\alpha = 0.90$) sẽ khiến mô hình mất "trí nhớ". |
| **H4: Tính Diễn giải (Explainability)** | Trích xuất thành công Heatmap và PCA các hệ số KAN nhạy cảm nhất. | Mở ra kỷ nguyên Explainable AI (XAI) cho Continual Learning trong Security. |
| **H5: Độ Phức tạp Tính toán** | Overhead = **0.007ms** / 10,000 samples. RAM cực nhỏ (69MB). | Phá vỡ rào cản triển khai thời gian thực trên các hệ thống mạng lớn (Backbone). |
| **H6: Khả năng Chịu lỗi & Tổng quát**| FAR = 0% dưới nhiễu Gaussian (1.0) và Nhiễu Nhãn (20%). Generalization plot ổn định. | Chứng minh mô hình học bản chất thực sự của gói tin, không bị overfitting hay ghi nhớ theo thứ tự dữ liệu. |

---

## 📂 Cấu trúc Thư mục

- `src/`: Chứa toàn bộ mã nguồn mô hình, data loader và các kịch bản H1-H6.
- `reports/`: Chứa toàn bộ biểu đồ, file đánh giá `.csv` của từng thực nghiệm.
- `logs/`: Chứa log chạy thực nghiệm chi tiết từng Epoch.
- `models/checkpoints/`: Lưu trữ các trọng số (Weights) `.pth` của mô hình ở mọi chu kỳ Drift.
- `TONG_QUAN_HE_THONG.md`: Tài liệu đặc tả kỹ thuật chi tiết nhất (Công thức toán, logic nội bộ).

---
**Tác giả & Đóng góp:** Phát triển cho hệ thống mạng CyberSecurity hiện đại. Mọi kết quả đã được Push đầy đủ lên Repository này.
