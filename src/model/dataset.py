import torch
from torch.utils.data import Dataset, WeightedRandomSampler
import pandas as pd
import numpy as np
from collections import Counter
import pyarrow.parquet as pq

class TimeSeriesFlowDataset(Dataset):
    """
    Biến dữ liệu thành Time Series 3D.
    Hỗ trợ đọc Sub-file Chunking và Tự động căn lề cột đa miền.
    """
    def __init__(self, parquet_path, seq_length=10, label_mapping=None, start_offset=0, chunk_size=None, feature_cols=None):
        table = pq.read_table(parquet_path)
        
        if chunk_size is not None:
            table = table.slice(start_offset, chunk_size)
            
        df = table.to_pandas()
        self.labels_text = df['Label'].values
        
        # --- BẢN VÁ: TỰ ĐỘNG CĂN LỀ CỘT ĐA MIỀN ---
        X_df = df.drop(columns=['Label'])
        if feature_cols is not None:
            missing_cols = set(feature_cols) - set(X_df.columns)
            for c in missing_cols:
                X_df[c] = 0.0
            X_df = X_df[feature_cols]
            
        self.features = X_df.values.astype(np.float32)
        
        if label_mapping is None:
            unique_labels = sorted(list(set(self.labels_text)))
            self.label_mapping = {label: idx for idx, label in enumerate(unique_labels)}
        else:
            self.label_mapping = label_mapping
            
        self.y = np.array([self.label_mapping.get(lbl, 0) for lbl in self.labels_text], dtype=np.int64)
        
        self.seq_length = seq_length
        self.num_samples = len(self.features) - self.seq_length + 1
        
    def __len__(self):
        return max(0, self.num_samples)
    
    def __getitem__(self, idx):
        window_x = self.features[idx : idx + self.seq_length]
        target_y = self.y[idx + self.seq_length - 1]
        return torch.tensor(window_x), torch.tensor(target_y)

def create_weighted_sampler(dataset):
    actual_labels = dataset.y[dataset.seq_length - 1:]
    class_counts = Counter(actual_labels)
    
    num_samples = len(actual_labels)
    class_weights = {cls: num_samples / count for cls, count in class_counts.items()}
    sample_weights = [class_weights[lbl] for lbl in actual_labels]
    
    sampler = WeightedRandomSampler(
        weights=torch.DoubleTensor(sample_weights),
        num_samples=num_samples,
        replacement=True
    )
    return sampler, dataset.label_mapping