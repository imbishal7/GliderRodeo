#!/usr/bin/env bash
# Foreground launcher: use nohup or tmux for a remote run; no global process kills.
set -euo pipefail
experiment_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$experiment_root"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export TABPFN_MODEL_CACHE_DIR="${TABPFN_MODEL_CACHE_DIR:-$HOME/.cache/tabpfn}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-$HOME/.cache/matplotlib}"
export HF_HUB_DISABLE_TELEMETRY=1
export DO_NOT_TRACK=1
export PYTHONUNBUFFERED=1
mkdir -p "$HF_HOME" "$TABPFN_MODEL_CACHE_DIR" "$MPLCONFIGDIR"
exec timeout --signal=TERM --kill-after=30s 10800 "$experiment_root/.venv/bin/python" -m experiments.run "$@"
