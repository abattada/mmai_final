from datasets import load_dataset
from PIL import Image
import os
import io
import json

# 轉出目的地
OUT_ROOT = "./magicbrush_converted"
IMG_DIR = os.path.join(OUT_ROOT, "images")
META_DIR = os.path.join(OUT_ROOT, "meta")

os.makedirs(IMG_DIR, exist_ok=True)
os.makedirs(META_DIR, exist_ok=True)

# ✅ 控制每個 split 要處理幾筆
#   - 先設成 1 方便測試
#   - 要整個 dataset 時，改成 None 就會全轉
MAX_SAMPLES_PER_SPLIT = None  # or None for all


def to_pil(img_feat):
    """
    把 MagicBrush 的 image 欄位統一轉成 PIL.Image
    可能是：
    - PIL.Image
    - dict: {'bytes': ..., 'path': ...}
    """
    if isinstance(img_feat, Image.Image):
        return img_feat

    if isinstance(img_feat, dict):
        if "bytes" in img_feat and isinstance(img_feat["bytes"], (bytes, bytearray)):
            return Image.open(io.BytesIO(img_feat["bytes"]))
        if "path" in img_feat:
            return Image.open(img_feat["path"])

    raise ValueError(f"Unknown image format: {type(img_feat)} - {img_feat}")


print("🚀 從本機 cache 載入完整 MagicBrush dataset（使用 ./data/osunlp__magic_brush）...")
ds_dict = load_dataset("osunlp/MagicBrush", cache_dir="./data")

for split_name, split_ds in ds_dict.items():
    print(f"📂 處理 split = {split_name}, 共 {len(split_ds)} 筆")

    count = 0
    for ex in split_ds:
        if count % 50 == 0:
            print(f"   處理中... 已處理 {count} 筆")
        img_id = ex["img_id"]
        turn_idx = ex["turn_index"]
        instr = ex["instruction"]

        base_name = f"{split_name}_{img_id}_turn{turn_idx}"

        src_path = os.path.join(IMG_DIR, f"{base_name}_source.png")
        mask_path = os.path.join(IMG_DIR, f"{base_name}_mask.png")
        tgt_path = os.path.join(IMG_DIR, f"{base_name}_target.png")
        meta_path = os.path.join(META_DIR, f"{base_name}.json")

        # ✅ 四個檔案都存在就跳過
        if (
            os.path.exists(src_path)
            and os.path.exists(mask_path)
            and os.path.exists(tgt_path)
            and os.path.exists(meta_path)
        ):
            print(f"⏭ 已存在，跳過: {base_name}")
            count += 1
            # 即使跳過也算在 count 裡，避免重複轉
            if MAX_SAMPLES_PER_SPLIT is not None and count >= MAX_SAMPLES_PER_SPLIT:
                break
            continue

        # 轉成 PIL
        src_img = to_pil(ex["source_img"]).convert("RGB")
        mask_img = to_pil(ex["mask_img"])  # mask 保留原格式
        tgt_img = to_pil(ex["target_img"]).convert("RGB")

        # 存圖片
        src_img.save(src_path)
        mask_img.save(mask_path)
        tgt_img.save(tgt_path)

        # 存 meta（包含 prompt / instruction）
        meta = {
            "img_id": img_id,
            "turn_index": turn_idx,
            "instruction": instr,  # 這就是你要的 prompt
            "source_path": os.path.relpath(src_path, OUT_ROOT),
            "mask_path": os.path.relpath(mask_path, OUT_ROOT),
            "target_path": os.path.relpath(tgt_path, OUT_ROOT),
            "split": split_name,
        }
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

        print(f"✅ 已輸出: {base_name}")
        count += 1

        # 只先轉前幾筆（目前是 1 筆）
        if MAX_SAMPLES_PER_SPLIT is not None and count >= MAX_SAMPLES_PER_SPLIT:
            break

print("🎉 MagicBrush 轉檔完成（目前每個 split 只轉前幾筆），輸出在：", OUT_ROOT)
