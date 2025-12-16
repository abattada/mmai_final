# 專案概觀

本倉庫整理了簡報編輯資料集的建立流程，以及以 PowerPaint 進行「已知 bbox（No-YOLO）」推論與評估的腳本。輸入來源包含 MagicBrush triplets（source/target/mask）與自製的簡報背景與 bounding boxes，輸出為可訓練/評估的 slide-level 影像與標註。

## 專案結構速覽
- `base_slide/`：1280×720 的簡報底圖；`bbox_slide/` 為帶有 bbox 標註的視覺化版本。
- `bboxes.json`：每張底圖的 bounding box 設定，鍵為頁面 ID，值為 `{"1": [x1, y1, x2, y2], ...}`。
- `magicbrush_converted/`：`download_magicbrush.py` 轉出的 MagicBrush 影像與 meta（`images/`、`meta/`）。
- `dataset/`、`gt/`：`build_slide.py` 產出的 slide-level 影像與對應的 `meta.json`。
- `requirements.txt`：推論與轉檔所需套件（torch/diffusers/transformers 等）。
- 主要腳本：`download_magicbrush.py`、`view_box.py`、`build_slide.py`、`fill_prompt.py`、`json_to_txt.py`、`no_yolo_inference.py`，以及測試腳本 `test_no_yolo.sh`、`test_power_paint.py`。

## 環境安裝
在虛擬環境中安裝依賴：
```bash
pip install -r requirements.txt
```

## 工作流程
以下步驟將 MagicBrush 資料轉成 slide-level 資料集，並進行 No-YOLO 推論。

### 1. 下載並整理 MagicBrush 資料
使用 `download_magicbrush.py` 透過 `datasets` 載入 `osunlp/MagicBrush`，轉存為圖片與 meta：
```bash
python download_magicbrush.py  # 輸出至 magicbrush_converted/images 與 magicbrush_converted/meta
```
檔名格式為 `<split>_<img_id>_turn<idx>_{source|mask|target}.png`，meta 會寫入對應的 instruction（prompt）。程式預設 `MAX_SAMPLES_PER_SPLIT = None` 轉出全部，可視需求在檔案內調整。

完成後可用 `count_images.py` 檢查轉檔數量與完整性：
```bash
python count_images.py --root ./magicbrush_converted
```

### 2. 建立/檢視 bounding boxes
`view_box.py` 協助在 `base_slide/<ID>.png` 上標註與檢查 bbox，結果存入 `bboxes.json`。
- 單次新增並輸出除錯圖：
  ```bash
  python view_box.py 011 100 200 300 350 -s  # 在 011.png 上新增一個 bbox 並存檔
  ```
- 檢視所有頁面的標註（畫在對應底圖上）：
  ```bash
  python view_box.py -v
  ```
除錯輸出會放在 `./test/debug_<ID>.png`，`-s/--save` 會將新框寫入 `bboxes.json`。

### 3. 產生 slide 編輯資料集
`build_slide.py` 根據 `bboxes.json` 與 MagicBrush turn1 圖片，將 patch 隨機貼入每個 bbox，產生 slide-level source/target/mask 以及 `gt/<split>/meta.json`。
```bash
python build_slide.py \
  --num 1 \          # 每個 MagicBrush patch 在每張底圖生成張數
  --workers 4        # 平行處理 thread 數
```
- 輸出影像：`dataset/train|validation|test/*.png`
- 標註：`gt/train|validation|test/meta.json`（含 `source`/`target`/`mask` 檔名、`bbox`、`prompt`）。
- 使用 `--test-run` 可快速產生最小樣本。

### 4. 補齊 prompt（若 meta 缺少 instruction）
`fill_prompt.py` 會讀取 `train_meta.json`、`dev_meta.json`，並從 `magicbrush_converted/meta/<split>_<id>_turn1.json` 補上缺漏的 `prompt` 欄位：
```bash
python fill_prompt.py
```

### 5. 轉成 YOLO bbox 標註
`json_to_txt.py` 會將 `gt/<split>/meta.json` 內的 `bbox` 轉換成 YOLO 格式，分別為 source/target/mask 各輸出一個 `.txt` 標註檔至 `gt/<split>/label/`：
```bash
python json_to_txt.py
```

### 6. No-YOLO Oracle 推論與評估
`no_yolo_inference.py` 會讀取 `gt/<split>/meta.json`，對每筆樣本以「已知 bbox」裁切成 patch，將 patch 與 mask 丟入 PowerPaint，並把生成結果貼回原圖。程式同時計算：
- **Masked LPIPS**：只針對遮罩範圍量測編輯強度。
- **CLIP score**：生成區塊與文字指令的相似度。
- **Background LPIPS**：非遮罩背景維持程度。
- **CLIP-I**：整張輸出與 GT 的影像相似度。

基本使用範例（對 validation split）：
```bash
python no_yolo_inference.py \
  --meta_path gt/validation/meta.json \
  --img_dir dataset/validation \
  --out_dir no_yolo_results/validation \
  --steps 30 \
  --guidance_scale 7.5 \
  --num_vis 8              # 隨機存 8 個可視化
```
- `--model_path <peft_adapter>` 可選擇載入 LoRA（需要已安裝 `peft`）。
- 會輸出 `results_no_yolo.json`（逐樣本結果）與 `summary_no_yolo.json`（平均/標準差），可視化檔案放在 `out_dir/vis/`。
- `--crop_size` 決定計算指標時重採樣的邊長；`--seed` 控制可視化抽樣的隨機性。

`shell` 腳本 `test_no_yolo.sh` 示範了設定環境變數並呼叫上述指令，可依需求調整 `SPLIT`/`MODEL_PATH`/推論參數後直接執行。

## 其他工具
- `test_power_paint.py`：示範結合 Grounding DINO + CLIPSeg + PowerPaint，在指定簡報截圖上自動找框與 inpaint，會輸出除錯框 (`debug_groundingdino_box.png`)、mask (`debug_clipseg_mask.png`) 與成品 (`final_result.png`)。
- `view_box.py -v` 生成的標註視覺化可協助人工檢查框位置是否合理。

