#!/bin/bash
# ============================================================
# Vast.ai cloud setup — environment check + install + train
# ============================================================
set -e

echo "============================================================"
echo "  Fragrance Fine-tuning — Cloud Setup"
echo "============================================================"

# ---- 1. Environment check ----
echo ""
echo "[1/4] Checking environment..."
echo "  Python: $(python --version)"
echo "  CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())' 2>/dev/null || echo 'NO')"
echo "  GPU: $(python -c 'import torch; print(torch.cuda.get_device_name(0))' 2>/dev/null || echo 'N/A')"
echo "  VRAM: $(python -c 'import torch; gb=torch.cuda.get_device_properties(0).total_memory/1024**3; print(f\"{gb:.1f} GB\")' 2>/dev/null || echo 'N/A')"

# ---- 2. Install dependencies ----
echo ""
echo "[2/4] Installing dependencies..."
pip install --upgrade pip -q
pip install unsloth trl datasets accelerate -q

# ---- 3. Verify install ----
echo ""
echo "[3/4] Verifying Unsloth..."
python -c 'from unsloth import FastLanguageModel; print("  Unsloth OK")'

# ---- 4. Train ----
echo ""
echo "[4/4] Starting training..."
echo "============================================================"
python finetune/train.py
echo "============================================================"

echo ""
echo "Done! Model saved at: finetune/fragrance-qwen-7b-final/"
echo "Download this entire folder before destroying the instance!"
