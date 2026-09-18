#!/usr/bin/env bash
set -euo pipefail
study_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$study_root"
study_stage="${1:?Specify experts or finetune}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export TABPFN_MODEL_CACHE_DIR="${TABPFN_MODEL_CACHE_DIR:-$HOME/.cache/tabpfn}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-$HOME/.cache/matplotlib}"
export HF_HUB_DISABLE_TELEMETRY=1 DO_NOT_TRACK=1 PYTHONUNBUFFERED=1
for study_seed in 0 1; do
  study_protocols=(logo forward spatial)
  if [ "$study_seed" = 1 ]; then study_protocols=(forward); fi
  study_output="out/experiments/followup-${study_stage}-seed${study_seed}-20260918"
  for study_attempt in 1 2; do
    timeout --signal=TERM --kill-after=30s 10800 .venv/bin/python -m experiments.followup.run \
      --stage "$study_stage" --seed "$study_seed" --protocols "${study_protocols[@]}" \
      --output "$study_output" --resume
    study_status="$(.venv/bin/python -c 'import json,sys; print(json.load(open(sys.argv[1]))["status"])' "$study_output/manifest.json")"
    if [ "$study_status" = complete ]; then break; fi
    if [ "$study_status" != budget_reached ]; then
      echo "Campaign failed: $study_output status=$study_status" >&2
      exit 1
    fi
  done
  if [ "$study_status" != complete ]; then
    echo "Campaign budget exhausted: $study_output" >&2
    exit 1
  fi
done
