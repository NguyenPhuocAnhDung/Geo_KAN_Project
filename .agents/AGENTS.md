
# Terminal Management Rule
KHÔNG chạy các tiến trình nặng hoặc có log trên terminal gốc nữa.
Sử dụng Tmux: Khi bắt đầu trích xuất dữ liệu, chạy pipeline hoặc bất kỳ tiến trình lâu dài nào, thao tác ĐẦU TIÊN là tạo một phiên bản tmux độc lập (Ví dụ: `tmux new -s pipeline_run`).
Quy trình chuẩn:
1. Vào tmux (`tmux new -s <name>`).
2. Chạy lệnh thực thi.
3. Tách ra ngoài (Detach) `tmux detach`. Lúc này, toàn bộ log và tiến trình sẽ bị "nhốt" an toàn trong session đó. Màn hình console của tài khoản quyhv sẽ luôn sạch sẽ, gọn gàng và không bao giờ bị rác ký tự nữa. AI cũng có thể chủ động chạy các lệnh trong tmux để tránh spam terminal của người dùng.
