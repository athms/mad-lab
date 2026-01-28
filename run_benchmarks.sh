#!/bin/bash
# MAD Benchmark Comparison Script
# Compares GDN/BS-GDN against Mamba and Gated Linear Attention baselines

set -e  # Exit on error

WANDB_PROJECT="MAD"

echo "=========================================="
echo "Starting MAD Benchmark Suite"
echo "=========================================="

# Model 1: Basis-Subspace Gated Delta Net
echo ""
echo "=== Running BS-Gated Delta Net ==="
python -m benchmark \
    --log-to-wandb \
    --wandb-project "$WANDB_PROJECT" \
    --no-save-checkpoints \
    --layers bs-gated-delta-net swiglu bs-gated-delta-net swiglu

# Model 2: Gated Delta Net
echo ""
echo "=== Running Gated Delta Net ==="
python -m benchmark \
    --log-to-wandb \
    --wandb-project "$WANDB_PROJECT" \
    --no-save-checkpoints \
    --layers gated-delta-net swiglu gated-delta-net swiglu

# Baseline 1: Mamba
echo ""
echo "=== Running Mamba baseline ==="
python -m benchmark \
    --log-to-wandb \
    --wandb-project "$WANDB_PROJECT" \
    --no-save-checkpoints \
    --layers mamba swiglu mamba swiglu

# Baseline 2: Gated Linear Attention
echo ""
echo "=== Running Gated Linear Attention baseline ==="
python -m benchmark \
    --log-to-wandb \
    --wandb-project "$WANDB_PROJECT" \
    --no-save-checkpoints \
    --layers gated-linear-attention swiglu gated-linear-attention swiglu

echo ""
echo "=========================================="
echo "All benchmarks completed!"
echo "=========================================="
