#!/usr/bin/env bash
set -euo pipefail

GPU="${GPU:-4}"
FOLDS="${FOLDS:-5}"
SEED="${SEED:-7}"
PROBE_LIMIT="${PROBE_LIMIT:-200}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-192}"
LATENT_MAX_TOKENS="${LATENT_MAX_TOKENS:-768}"
CONDA_ENV="${CONDA_ENV:-/data1/JustinLu090/.cache/conda/envs/latentmem}"
MODEL="${MODEL:-Qwen/Qwen3-4B-Instruct-2507}"

INPUT_JSONL="${INPUT_JSONL:-data/trajectories/popqa2500_trajectory_context_splits/popqa2500_traj_context_all.jsonl}"
DATA_ROOT="${DATA_ROOT:-data/trajectory_context_popqa2500_qwen3_4b_768}"
ALL_DIR="${ALL_DIR:-${DATA_ROOT}/all}"
FOLDS_DIR="${FOLDS_DIR:-${DATA_ROOT}/folds}"
CONFIG="${CONFIG:-configs/ssamem_trajectory_context_popqa2500_768_b4.yaml}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/trajectory_context_popqa2500_768_b4_cv}"

run_ssamem() {
  CUDA_VISIBLE_DEVICES="${GPU}" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    conda run --no-capture-output -p "${CONDA_ENV}" python -m ssamem "$@"
}

mkdir -p "${OUTPUT_ROOT}"

echo "Using GPU=${GPU}"
echo "Using conda env=${CONDA_ENV}"
echo "Using input=${INPUT_JSONL}"

if [[ ! -f "${ALL_DIR}/manifest.jsonl" ]]; then
  echo "===== Build all latent dataset ====="
  run_ssamem build-ssa-data \
    --input "${INPUT_JSONL}" \
    --output "${ALL_DIR}" \
    --runtime-mode hf \
    --model-name-or-path "${MODEL}" \
    --trust-remote-code \
    --device cuda \
    --torch-dtype float16 \
    --load-in-4bit \
    --latent-max-tokens "${LATENT_MAX_TOKENS}"
else
  echo "===== Skip build: ${ALL_DIR}/manifest.jsonl exists ====="
fi

if [[ ! -f "${FOLDS_DIR}/fold_00/train.jsonl" ]]; then
  echo "===== Create ${FOLDS}-fold manifests ====="
  conda run -p "${CONDA_ENV}" python -m ssamem kfold-ssa-manifest \
    --manifest "${ALL_DIR}/manifest.jsonl" \
    --output "${FOLDS_DIR}" \
    --folds "${FOLDS}" \
    --seed "${SEED}"
else
  echo "===== Skip kfold: ${FOLDS_DIR}/fold_00/train.jsonl exists ====="
fi

for ((i = 0; i < FOLDS; i++)); do
  FOLD="$(printf "%02d" "${i}")"
  FOLD_DIR="${OUTPUT_ROOT}/fold_${FOLD}"
  TRAIN_MANIFEST="${FOLDS_DIR}/fold_${FOLD}/train.jsonl"
  VAL_MANIFEST="${FOLDS_DIR}/fold_${FOLD}/val.jsonl"
  CHECKPOINT="${FOLD_DIR}/distiller.pt"
  HISTORY="${FOLD_DIR}/history.json"
  PROBE="${FOLD_DIR}/valid_probe.json"

  mkdir -p "${FOLD_DIR}"

  if [[ ! -f "${CHECKPOINT}" ]]; then
    echo "===== Train fold_${FOLD} ====="
    run_ssamem train-ssa \
      --config "${CONFIG}" \
      --manifest "${TRAIN_MANIFEST}" \
      --output-path "${CHECKPOINT}" \
      --history-path "${HISTORY}" \
      --device cuda \
      --run-name "popqa2500-768-b4-fold${FOLD}"
  else
    echo "===== Skip train fold_${FOLD}: checkpoint exists ====="
  fi

  if [[ ! -f "${PROBE}" ]]; then
    echo "===== Probe fold_${FOLD} ====="
    run_ssamem probe-ssa-generation \
      --config "${CONFIG}" \
      --manifest "${VAL_MANIFEST}" \
      --checkpoint "${CHECKPOINT}" \
      --device cuda \
      --limit "${PROBE_LIMIT}" \
      --max-new-tokens "${MAX_NEW_TOKENS}" \
      --output-path "${PROBE}"
  else
    echo "===== Skip probe fold_${FOLD}: probe exists ====="
  fi
done

SUMMARY="${OUTPUT_ROOT}/cv_summary.json"
echo "===== Summarize folds ====="
export OUTPUT_ROOT
python - <<'PY'
import json
import os
from pathlib import Path

root = Path(os.environ["OUTPUT_ROOT"])
rows = []
for fold_dir in sorted(root.glob("fold_*")):
    probe_path = fold_dir / "valid_probe.json"
    if not probe_path.exists():
        continue
    data = json.loads(probe_path.read_text())
    rows.append(
        {
            "fold": fold_dir.name,
            "no_memory": data["target_hit"]["no_memory"]["mean"],
            "text_memory": data["target_hit"]["text_memory"]["mean"],
            "latent_memory": data["target_hit"]["latent_memory"]["mean"],
            "probe_path": str(probe_path),
        }
    )

summary = {"folds": rows, "means": {}}
for key in ["no_memory", "text_memory", "latent_memory"]:
    vals = [row[key] for row in rows]
    if vals:
        summary["means"][key] = sum(vals) / len(vals)

out = root / "cv_summary.json"
out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(summary, ensure_ascii=False, indent=2))
PY

echo "Summary written to ${SUMMARY}"
