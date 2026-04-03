Geo KAN Project
Ask DeepWiki

This repository contains a research project focused on detecting Concept Drift in IoT and network data streams using a novel Hybrid Temporal Kolmogorov-Arnold Network (Hybrid TKAN) architecture.

The project provides a complete pipeline from data preprocessing of various network traffic datasets to training and evaluating a deep learning model designed to identify diverse cyber attacks.

Key Features
Hybrid TKAN Model: Implements a unique architecture combining an LSTM layer to capture temporal features, an MLP for feature transformation, and a custom ChebyshevKANLayer for classification. This leverages the strengths of Kolmogorov-Arnold Networks for potentially better accuracy and interpretability.
Robust Data Preprocessing: A comprehensive script (data_preprocess.py) that unifies multiple network traffic datasets (e.g., CIC-IDS-2017/2018, CICDDoS2019, CICIoT2023) by:
Standardizing column names across different schemas.
Mapping dozens of specific attack labels into 8 primary categories (Benign, DDoS, DoS, Botnet, Web_Attack, Brute_Force, PortScan, Infiltration).
Cleaning data, handling infinite/NaN values, and preserving time-series integrity.
Saving processed data efficiently in Parquet format.
Global Feature Scaling: A dedicated script (build_global_scaler.py) creates a single StandardScaler object across the entire training dataset. This ensures consistent feature scaling for all data, preventing data leakage and making the model robust to datasets with varying feature sets.
Optimized Training Pipelines: Two distinct training scripts are provided:
train_phase1.py: Optimized for GPU-accelerated training, leveraging CUDA, pin_memory, and non-blocking data transfers for maximum performance.
train_phase1_ram.py: Tailored for high-RAM, multi-core CPU servers, utilizing a large number of workers for parallel data loading.
Advanced Training Techniques: The training scripts incorporate best practices such as class weighting for imbalanced datasets, gradient clipping, AdamW optimizer, learning rate scheduling (ReduceLROnPlateau), and early stopping to prevent overfitting.
Model Architecture
The Hybrid_TKAN model is a sequential network composed of three main blocks:

LSTM Layer (nn.LSTM): The first layer processes the input time-series data (sequences of network flow statistics) to capture temporal dependencies and patterns over time.
MLP Block (nn.Sequential): The output from the LSTM is passed through a Multi-Layer Perceptron consisting of Linear layers, LayerNorm, and GELU activation functions. This block further processes and refines the learned temporal features.
Chebyshev KAN Layer (ChebyshevKANLayer): The final classification is performed by a custom Kolmogorov-Arnold Network layer. Instead of a simple linear transformation, this layer uses a basis of Chebyshev polynomials to learn complex activation functions, potentially leading to better modeling of intricate relationships in the data.
Project Structure
.
├── models/
│   ├── checkpoints/         # Saved model weights (.pth files)
│   └── global_scaler.pkl   # The global feature scaler
├── reports/
│   └── plots/              # Saved plots, like confusion matrices
├── src/
│   ├── data_preprocess/
│   │   └── data_preprocess.py  # Script to clean and unify raw datasets
│   ├── model/
│   │   ├── dataset.py        # PyTorch Dataset classes
│   │   ├── kan_layer.py      # Custom Chebyshev KAN Layer implementation
│   │   └── model.py          # Hybrid_TKAN model definition
│   ├── build_global_scaler.py  # Script to create the global scaler
│   ├── train_phase1.py         # Training script optimized for GPU
│   └── train_phase1_ram.py     # Training script optimized for high-RAM CPUs
└── requirements.txt            # Project dependencies
Getting Started
1. Installation
Clone the repository and install the required dependencies.

git clone https://github.com/NguyenPhuocAnhDung/Geo_KAN_Project.git
cd Geo_KAN_Project
pip install -r requirements.txt
2. Data Preparation
Place your raw network traffic datasets (in .csv or .parquet format) into the dataset/raw/ directory. The preprocessing script expects a structure like dataset/raw/<DatasetName>/<files...>.
Run the data preprocessing script. This will process all datasets defined in DATASET_PHASES, standardize them, and save the cleaned output to dataset/processed/.
python src/data_preprocess/data_preprocess.py
3. Build Global Scaler
After preprocessing the data, create the global feature scaler. This script scans all processed training files to create a single, unified scaler that accounts for all possible features.

python src/build_global_scaler.py
This will create the models/global_scaler.pkl file, which is required for training.

4. Training
You can choose the training script based on your available hardware. Model checkpoints, logs, and evaluation plots will be saved to the models/checkpoints/, logs/, and reports/plots/ directories, respectively.

For GPU-based training:

python src/train_phase1.py
For high-RAM CPU-based training:

python src/train_phase1_ram.py
Dependencies
All required Python packages are listed in the requirements.txt file. The core libraries include:

torch
pandas
scikit-learn
numpy
pyarrow
imbalanced-learn
matplotlib & seaborn
