import torch
import numpy as np
from collections import defaultdict
from torch.utils.data import Dataset, DataLoader

class ClassBalancedReservoirBuffer:
    def __init__(self, max_size_per_class=500):
        """
        Khởi tạo Class-Balanced Reservoir Sampling (CBRS) Buffer.
        
        Chiến lược Sampling: 
        Dữ liệu mạng có tính Long-tail cực đoan. Nếu dùng Random Sampling, nhãn thiểu số sẽ bị xóa sổ. 
        CBRS chia không gian bộ nhớ M thành C phân vùng độc lập cho C class. Khi mẫu mới của class c đến, 
        nó chỉ thực hiện thuật toán Reservoir cục bộ trong phân vùng của class c. 
        Cấu trúc này có Time Complexity O(1) và Space Complexity O(M) (Bounded memory), 
        đảm bảo bảo tồn vĩnh viễn ký ức về các cuộc tấn công hiếm.
        
        max_size_per_class: Số lượng mẫu tối đa được lưu trữ cho mỗi nhãn.
        """
        self.max_size_per_class = max_size_per_class
        # Dictionary lưu trữ dữ liệu theo từng class: class_id -> list of samples
        self.buffer_x = defaultdict(list)
        self.buffer_y = defaultdict(list)
        # Dictionary đếm tổng số mẫu của mỗi class đã đi qua luồng (dùng cho Reservoir Sampling)
        self.samples_seen_per_class = defaultdict(int)

    def add_samples(self, x, y):
        """
        Thêm một batch dữ liệu (x, y) vào buffer sử dụng thuật toán Reservoir Sampling độc lập.
        x: tensor có shape (Batch_size, Features)
        y: tensor có shape (Batch_size,)
        """
        x_np = x.detach().cpu().numpy()
        y_np = y.detach().cpu().numpy()

        for i in range(len(y_np)):
            label = int(y_np[i])
            self.samples_seen_per_class[label] += 1
            
            # Nếu buffer của class này chưa đầy, thêm trực tiếp vào
            if len(self.buffer_y[label]) < self.max_size_per_class:
                self.buffer_x[label].append(x_np[i])
                self.buffer_y[label].append(label)
            else:
                # Nếu đã đầy, bốc ngẫu nhiên một index để thay thế (Reservoir Sampling)
                # Xác suất thay thế = max_size / số mẫu đã thấy
                j = np.random.randint(0, self.samples_seen_per_class[label])
                if j < self.max_size_per_class:
                    self.buffer_x[label][j] = x_np[i]

    def get_balanced_batch(self, batch_size):
        """
        Lấy ra một batch dữ liệu cân bằng từ Buffer để trộn với dữ liệu mới.
        Nếu số mẫu trong buffer ít hơn batch_size, sẽ lấy toàn bộ.
        """
        if len(self.buffer_y) == 0:
            return None, None
            
        # Chia đều số lượng lấy ra cho mỗi class
        num_classes = len(self.buffer_y)
        samples_per_class = max(1, batch_size // num_classes)
        
        batch_x = []
        batch_y = []
        
        for label in self.buffer_y.keys():
            current_size = len(self.buffer_y[label])
            if current_size == 0:
                continue
                
            # Lấy ngẫu nhiên các mẫu từ class này
            indices = np.random.choice(
                current_size, 
                min(samples_per_class, current_size), 
                replace=False
            )
            
            for idx in indices:
                batch_x.append(self.buffer_x[label][idx])
                batch_y.append(self.buffer_y[label][idx])
                
        if len(batch_x) == 0:
            return None, None
            
        # Shuffle lại mảng trước khi trả về
        combined = list(zip(batch_x, batch_y))
        np.random.shuffle(combined)
        batch_x, batch_y = zip(*combined)
        
        return torch.tensor(np.array(batch_x), dtype=torch.float32), torch.tensor(np.array(batch_y), dtype=torch.long)

class StreamDataset(Dataset):
    """Lớp bọc cho việc mix dữ liệu từ luồng mới và Replay Buffer."""
    def __init__(self, x_stream, y_stream, x_buffer=None, y_buffer=None):
        self.x = x_stream
        self.y = y_stream
        if x_buffer is not None and y_buffer is not None:
            self.x = torch.cat([self.x, x_buffer], dim=0)
            self.y = torch.cat([self.y, y_buffer], dim=0)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]
