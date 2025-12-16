import torch
from transformers import (
    CLIPSegProcessor,
    CLIPSegForImageSegmentation,
    AutoProcessor,
    AutoModelForZeroShotObjectDetection,
)

import sys
import huggingface_hub as _hf
from huggingface_hub import hf_hub_download as _hf_hub_download

if not hasattr(_hf, "cached_download"):
    def cached_download(*args, **kwargs):
        return _hf_hub_download(*args, **kwargs)

    _hf.cached_download = cached_download
    sys.modules["huggingface_hub"] = _hf
# ========================================================================

from diffusers import StableDiffusionInpaintPipeline, DDPMScheduler
from diffusers.utils import load_image
from PIL import Image, ImageDraw
import numpy as np

# ==========================
# 0. 參數設定
# ==========================
IMG_PATH = "Screenshot 2025-11-27 200322.png"           # 你的簡報截圖
BOX_TO_FIND = "table" # Grounding DINO 搜尋文字
TEXT_TO_FIND = "table"   # CLIPSeg 搜尋文字
PROMPT = "change the table for a dog"  # PowerPaint 要畫什麼
DEVICE = "cuda"

GDINO_MODEL_ID = "IDEA-Research/grounding-dino-tiny"  # Grounding DINO 模型

print(f"🚀 初始化... 正在載入 CLIPSeg、Grounding DINO 和 PowerPaint...")

# ------------------------------------------------
# 載入 CLIPSeg (負責 segmentation)
# ------------------------------------------------
clipseg_processor = CLIPSegProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")
clipseg_model = CLIPSegForImageSegmentation.from_pretrained(
    "CIDAS/clipseg-rd64-refined"
).to(DEVICE)

# ------------------------------------------------
# 載入 Grounding DINO (負責 bounding box)
# ------------------------------------------------
gdino_processor = AutoProcessor.from_pretrained(GDINO_MODEL_ID)
gdino_model = AutoModelForZeroShotObjectDetection.from_pretrained(
    GDINO_MODEL_ID
).to(DEVICE)

# ------------------------------------------------
# 載入 PowerPaint (負責修圖 / inpaint)
# ------------------------------------------------
pipe = StableDiffusionInpaintPipeline.from_pretrained(
    "Sanster/PowerPaint-V1-stable-diffusion-inpainting",
    torch_dtype=torch.float16,
    safety_checker=None,
).to(DEVICE)

# ==========================
# 1. 讀圖
# ==========================
original_image = load_image(IMG_PATH).convert("RGB")
W, H = original_image.size
print(f"📸 讀取圖片: {W}x{H}")

# ==========================
# 2. Grounding DINO 找 bounding box
# ==========================
print(f"🧭 Grounding DINO 正在尋找 '{BOX_TO_FIND}' 的區域 (bounding box)...")

# ✅ 這裡改成字串（或 List[str]），不要用 List[List[str]]
gdino_text = BOX_TO_FIND

gdino_inputs = gdino_processor(
    images=original_image,
    text=gdino_text,
    return_tensors="pt",
).to(DEVICE)

with torch.no_grad():
    gdino_outputs = gdino_model(**gdino_inputs)

# 把 raw output 轉成實際座標 (x0, y0, x1, y1)
gdino_results = gdino_processor.post_process_grounded_object_detection(
    outputs=gdino_outputs,
    input_ids=gdino_inputs.input_ids,
    box_threshold=0.09,      # box confidence 門檻
    text_threshold=0.05,     # 文字匹配門檻
    target_sizes=[(H, W)],   # (height, width)
)

gdino_res = gdino_results[0]
boxes = gdino_res["boxes"]   # tensor [num_boxes, 4]
scores = gdino_res["scores"] # tensor [num_boxes]

if boxes.shape[0] == 0:
    raise RuntimeError(
        f"[Grounding DINO] 沒有找到和 '{BOX_TO_FIND}' 對應的物件，"
        "可以試試降低 threshold 或換一個描述。"
    )

# 取分數最高的那個 box
best_idx = scores.argmax().item()
best_box = boxes[best_idx].tolist()
x0, y0, x1, y1 = [int(v) for v in best_box]
print(f"✅ Grounding DINO 最佳框: ({x0}, {y0}) -> ({x1}, {y1}), score = {scores[best_idx].item():.3f}")

# 輸出畫有 bounding box 的 debug 圖
debug_box_img = original_image.copy()
draw = ImageDraw.Draw(debug_box_img)
draw.rectangle([x0, y0, x1, y1], outline="red", width=3)
draw.text((x0, max(0, y0 - 15)), BOX_TO_FIND, fill="red")
debug_box_img.save("debug_groundingdino_box.png")
print("🖼 已輸出 Grounding DINO 結果圖：debug_groundingdino_box.png")

# ==========================
# 3. 在 bounding box 裡用 CLIPSeg 做 mask
# ==========================
print(f"🎯 CLIPSeg 在 box 中尋找 '{TEXT_TO_FIND}' 的精細 mask...")

# 裁出 Grounding DINO 提供的子圖
crop = original_image.crop((x0, y0, x1, y1))
box_w, box_h = crop.size

clipseg_inputs = clipseg_processor(
    text=[TEXT_TO_FIND],
    images=[crop],
    return_tensors="pt",
).to(DEVICE)

with torch.no_grad():
    clipseg_outputs = clipseg_model(**clipseg_inputs)

# logits 形狀為 (batch, H', W') → 取第 0 張
preds = torch.sigmoid(clipseg_outputs.logits)[0]  # (h', w')
mask_crop = preds.cpu().numpy()

# 二值化
mask_crop = (mask_crop > 0.4).astype(np.uint8) * 255  # 0 or 255

# 轉成與 bounding box 一樣大小
mask_crop_img = Image.fromarray(mask_crop).resize(
    (box_w, box_h), resample=Image.NEAREST
)

# 把 box 內的 mask 貼回整張圖上
full_mask_np = np.zeros((H, W), dtype=np.uint8)
full_mask_np[y0:y1, x0:x1] = np.array(mask_crop_img)

# [防禦機制] 強制把左邊 30% 塗黑，保護簡報左側文字
full_mask_np[:, : int(W * 0.3)] = 0

mask_image = Image.fromarray(full_mask_np)

# 輸出 CLIPSeg 得到的 mask debug 圖
mask_image.save("debug_clipseg_mask.png")
print("🖼 已輸出 CLIPSeg mask 圖：debug_clipseg_mask.png")

# ==========================
# 4. 只對 bounding box 區域做縮放，丟給 PowerPaint
# ==========================
# 這裡用的是 DINO 找到的 crop，而不是整張 original_image
patch_image = original_image.crop((x0, y0, x1, y1))      # bbox 裡的原圖
patch_mask = mask_image.crop((x0, y0, x1, y1))          # bbox 裡的 mask
box_w, box_h = patch_image.size

process_size = (512, 512)
input_image = patch_image.resize(process_size, resample=Image.LANCZOS)
input_mask = patch_mask.resize(process_size, resample=Image.NEAREST)

# ==========================
# 5. PowerPaint 推論（只看 bbox 這塊）
# ==========================
print(f"🎨 PowerPaint 正在繪製: '{PROMPT}'（只在 bounding box 區域）...")
output_small = pipe(
    prompt=PROMPT,
    image=input_image,
    mask_image=input_mask,
    negative_prompt="photorealistic, text, watermark, bad quality, blurry",
    num_inference_steps=50,
    strength=0.99,      # 接近 1.0 代表完全重繪 Mask 區域
    guidance_scale=12.5 # PowerPaint 建議用高一點的引導值
).images[0]

# 還原回 bounding box 原尺寸
output_patch = output_small.resize((box_w, box_h), resample=Image.LANCZOS)

# 只在 bbox 裡、且 mask 為白色的區域做替換，其餘保持原 crop
bbox_result = Image.composite(output_patch, patch_image, patch_mask)

# ==========================
# 6. 把處理好的 bbox 貼回整張圖
# ==========================
print("🔧 合成回高解析度原圖（僅替換 bounding box 區域）...")
final_image = original_image.copy()
final_image.paste(bbox_result, (x0, y0))

final_image.save("final_result.png")
print("🎉 大功告成！請查看：final_result.png、debug_groundingdino_box.png、debug_clipseg_mask.png")

