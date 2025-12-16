#!/bin/bash

# ============================================
# PowerPaint LoRA Training Launcher
# ============================================

# 1. GPU 設定
export CUDA_VISIBLE_DEVICES=0

# 2. 實驗名稱與時間
EXP_NAME="v4_slide_final"
DATE_STR=$(date +%Y%m%d_%H%M)

# 3. 關鍵路徑設定 (依照你的環境修改這裡)
META_FILE="data/meta/train_meta.json"
INPUT_DIR="data/images"                    # ⬅️ NEW: 圖片的根目錄
OUTPUT_DIR="outputs/${EXP_NAME}_${DATE_STR}"
LOG_DIR="runs/${EXP_NAME}_${DATE_STR}"

# 4. 訓練超參數
BATCH_SIZE=2
LR=1e-4
EPOCHS=50
IMAGE_SIZE=512
LORA_RANK=8

echo "========================================================="
echo "Starting Training: $EXP_NAME"
echo "Meta File:        $META_FILE"
echo "Image Root:       $INPUT_DIR"
echo "Output Directory: $OUTPUT_DIR"
echo "========================================================="

# 5. 執行 Python (記得傳入 --input_dir)
python train_powerpaint_mask.py \
    --meta_path "$META_FILE" \
    --input_dir "$INPUT_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --log_dir "$LOG_DIR" \
    --batch_size $BATCH_SIZE \
    --lr $LR \
    --epochs $EPOCHS \
    --image_size $IMAGE_SIZE \
    --lora_rank $LORA_RANK \
    --visualize_every 500 \
    --print_every 20 \
    --num_workers 4 \
    --max_train_time 21600

echo "Training finished!"