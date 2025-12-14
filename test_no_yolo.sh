#!/bin/bash
# ============================================================
# test_no_yolo.sh
# ============================================================

# 專案根目錄
ROOT_DIR=/local/abat/mmai/mmai_final

SPLIT=validation   # 想測 train / test 就改這個

META_PATH=${ROOT_DIR}/gt/${SPLIT}/meta.json
IMG_DIR=${ROOT_DIR}/dataset/${SPLIT}
OUT_DIR=${ROOT_DIR}/no_yolo_results/${SPLIT}

# -------------------------
# Model configuration
# -------------------------
# 如果不要 LoRA，就留空字串
MODEL_PATH=""

# -------------------------
# Inference configuration
# -------------------------
STEPS=1
GUIDANCE_SCALE=7.5

# ============================================================
# Run inference
# ============================================================

cd "${ROOT_DIR}" || exit 1

python no_yolo_inference.py \
  --meta_path "${META_PATH}" \
  --img_dir "${IMG_DIR}" \
  --out_dir "${OUT_DIR}" \
  --steps ${STEPS} \
  --guidance_scale ${GUIDANCE_SCALE} \
  --model_path "${MODEL_PATH}"
