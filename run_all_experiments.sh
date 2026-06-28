#!/bin/bash
# Script chạy toàn bộ pipeline thực nghiệm

# Kích hoạt môi trường ảo
source venv/bin/activate

echo "=========================================================="
echo "BẮT ĐẦU CHẠY TOÀN BỘ THỰC NGHIỆM TIFS Q1"
echo "=========================================================="

echo "[1/7] Chạy train_stream.py (Phase 1 Baseline & Thresholding)..."
venv/bin/python src/train_stream.py > logs/run_train_stream.log 2>&1
echo "✅ train_stream.py hoàn tất."

echo "[2/7] Chạy H1: Correlation Experiment..."
venv/bin/python src/experiment_correlation.py > logs/run_h1_correlation.log 2>&1
echo "✅ H1 hoàn tất."

echo "[3/7] Chạy H2: Latency Baselines Experiment..."
venv/bin/python src/experiment_latency_baselines.py > logs/run_h2_latency.log 2>&1
echo "✅ H2 hoàn tất."

echo "[4/7] Chạy H3: Ablation Experiment..."
venv/bin/python src/experiment_ablation.py > logs/run_h3_ablation.log 2>&1
echo "✅ H3 hoàn tất."

echo "[5/7] Chạy H4: Explainability Experiment..."
venv/bin/python src/experiment_explainability.py > logs/run_h4_explainability.log 2>&1
echo "✅ H4 hoàn tất."

echo "[6/7] Chạy H5: Complexity Experiment..."
venv/bin/python src/experiment_complexity.py > logs/run_h5_complexity.log 2>&1
echo "✅ H5 hoàn tất."

echo "[7/7] Chạy H6: Failure & Generalization Experiment..."
venv/bin/python src/experiment_failure_generalization.py > logs/run_h6_failure.log 2>&1
echo "✅ H6 hoàn tất."

echo "=========================================================="
echo "🎉 TOÀN BỘ THỰC NGHIỆM ĐÃ HOÀN TẤT!"
echo "=========================================================="
