# 專案程式與檔案用途說明

## bboxes.json
- **類型**：標註設定檔（JSON）
- **用途**：儲存每一張簡報底圖（base slide）的 bounding box 設定。
- **結構概念**：
  - 第一層 key：頁面 ID（例如 `"001"`）
  - 第二層 key：此頁面上第幾個框（`"1"`, `"2"`…）
  - value：`[x1, y1, x2, y2]`，對應 1280×720 圖片座標。
- **來源**：由 `view_box.py` 使用 `-s` 參數時寫入／累積。

---

## view_box.py
- **用途**：在簡報底圖上視覺化與管理 bounding box，並寫入 `bboxes.json`。
- **兩大功能**：

  1. **單次標註與存檔**
     - 輸入：
       - 圖片 ID（只給數字，例如 `011`，腳本會讀 `base_slide/011.png`）
       - 四個座標 `x1 y1 x2 y2`
     - 行為：
       - 在該圖上畫出新的紅色 bbox。
       - 同時把同一頁已存在於 `bboxes.json` 的 bbox 以綠色畫出。
       - 輸出 debug 圖到 `./test/debug_<ID>.png`。
       - 若加上 `-s / --save`，會將這個 bbox 寫入 `bboxes.json`，在該頁下一個可用 index 底下新增。

  2. **檢視所有已標註 bbox（-v 模式）**
     - 讀取 `bboxes.json`，把所有頁面與其 bounding boxes 畫在對應的 `base_slide/<ID>.png` 上（通常用綠色框）。
     - 可用於快速檢查目前哪些頁面已標註、框的位置是否合理。

- **典型用途**：
  - 手動標註簡報中可以放 MagicBrush 小圖的位置。
  - 視覺化檢查 `bboxes.json` 是否正確，避免貼圖跑版。
  
---

# Slide Editing Dataset & Evaluation Scripts

本文件說明兩個核心腳本：

- `build_slide.py`：從 MagicBrush triplets + 自製簡報背景 + bbox 配置，建出完整的 slide 編輯資料集。
- `no_yolo_inference.py`：在「已知 bbox（oracle）」的設定下，使用 PowerPaint 做推論與評估，不需要 YOLO。

---

## 1. `build_slide.py`：產生 slide 編輯資料集

### 1.1 功能概述

`build_slide.py` 會：

1. 讀取簡報底圖 `base_slide/*.png`（例如 001.png ~ 075.png）。
2. 讀取 `bboxes.json` 中，每張簡報上可以放圖片的 bounding boxes。
3. 讀取 `magicbrush_converted/images/` 中的 MagicBrush turn1 圖片 triplets：
   - `train_<id>_turn1_source.png`
   - `train_<id>_turn1_target.png`
   - `train_<id>_turn1_mask.png`
   - `dev_<id>_turn1_*` 也是同樣格式
4. 依照規則，把 MagicBrush 的 patch 隨機貼到簡報上的 bbox 裡，產生：
   - slide-level **source**（原圖 + 貼上去之前的 patch）
   - slide-level **target**（原圖 + 編輯後的 patch）
   - slide-level **mask**（僅標出可編輯區域）
5. 依照 split 規則切成 train / validation / test，並輸出：
   - 影像：`dataset/<train|validation|test>/*.png`
   - 標註：`gt/<train|validation|test>/meta.json`

---

### 1.2 依賴的檔案與目錄結構

執行前應確保以下結構存在（路徑可依實際專案根目錄修正）：

```text
.
├── base_slide/
│   ├── 001.png
│   ├── 002.png
│   ├── ...
│   └── 075.png
├── magicbrush_converted/
│   ├── images/
│   │   ├── train_XXXXX_turn1_source.png
│   │   ├── train_XXXXX_turn1_target.png
│   │   ├── train_XXXXX_turn1_mask.png
│   │   ├── dev_YYYYY_turn1_source.png
│   │   ├── ...
│   └── meta/
│       ├── train_XXXXX_turn1.json
│       ├── dev_YYYYY_turn1.json
├── bboxes.json
├── build_slide.py
└── （執行後產出）
    ├── dataset/
    │   ├── train/
    │   ├── validation/
    │   └── test/
    └── gt/
        ├── train/meta.json
        ├── validation/meta.json
        └── test/meta.json


---

## count_images.py
- **用途**：統計專案中各影像資料夾的圖片數量，作為資料檢查用的小工具。
- **典型行為**（依實作而定，預期功能）：
  - 計算例如：
    - `magicbrush_converted/images/` 下 train/dev 各自的 source/target/mask 數量。
    - `complete_slide/train` 與 `complete_slide/dev` 中圖檔數量。
  - 在終端機印出每個目錄與對應的檔案數，幫助確認轉檔是否完整。

---

## download_magicbrush.py
- **用途**：下載並整理 MagicBrush 資料集。
- **主要功能**：
  - 使用 `datasets` 的 `load_dataset("osunlp/MagicBrush", ...)` 載入資料。
  - 將其中的 `source_img` / `target_img` / `mask_img` 轉成實際的 `.png` 圖檔。
  - 搭配對應的 meta（包含 `instruction`），存到：
    - `magicbrush_converted/images/`
    - `magicbrush_converted/meta/`
- **備註**：
  - 初期版本可能只處理 train 前幾筆樣本，之後擴充成可轉整個 train/dev split。

---

## fill_prompt.py
- **用途**：補齊 `train_meta.json` / `dev_meta.json` 中缺少 `prompt` 欄位的樣本。
- **輸入**：
  - `train_meta.json`、`dev_meta.json`
  - `magicbrush_converted/meta/<split>_<id>_turn1.json`
- **邏輯**：
  - 逐一掃過 meta 中的 `samples`：
    - 若 sample 已有 `prompt` 且非空字串 → 保留原值。
    - 若沒有 `prompt`：
      - 由 `source` 檔名解析出 MagicBrush 圖片 ID。
      - 讀取對應的 `magicbrush_converted/meta/<split>_<id>_turn1.json`。
      - 將裡面的 `"instruction"` 填入 `prompt` 欄位。
  - 最後覆寫更新後的 meta 檔案。

---

## json_to_txt.py
- **用途**：把 `train_meta.json` / `dev_meta.json` 裡的 bbox 標註轉成 YOLO 格式的 `.txt` 標註檔。
- **輸入**：
  - `train_meta.json`、`dev_meta.json`
  - 對應的影像資料夾：
    - `complete_slide/train/`
    - `complete_slide/dev/`
- **輸出**：
  - `yolo_labels/train/*.txt`
  - `yolo_labels/dev/*.txt`
  - 每張圖一個 `.txt`，檔名與影像檔名相同（副檔名改 `.txt`）。
- **YOLO 格式**：
  - 單一類別（預設 class id = 0）：
    - `0 x_center y_center width height`
  - 所有座標均為相對值（0–1），由原始 bbox `[x1, y1, x2, y2]` 及圖片尺寸（1280×720）換算而來。
- **可調整項目**（依實作參數）：
  - 使用 `target` 或 `source` 當 YOLO 的影像來源。
  - 類別編號（class id）。

---

## requirements.txt
- **用途**：記錄專案所需的 Python 套件與版本，方便建立一致的虛擬環境。
- **內容大致包含**（實際以檔案為準）：
  - `torch`, `torchvision`
  - `transformers`
  - `diffusers`
  - `accelerate`
  - `safetensors`
  - `huggingface-hub`
  - `Pillow`
  - `numpy`
  - `datasets`
  - 以及其他通用工具套件（如 `tqdm`, `requests` 等）。
- **使用方式**：
  - 在虛擬環境中執行：
    ```bash
    pip install -r requirements.txt
    ```

---

## test_power_paint.py
- **用途**：實驗用腳本，將簡報截圖中的指定區域自動找出並用 PowerPaint 做 inpainting。
- **主要流程**：
  1. 讀取一張輸入圖片（簡報截圖，例如 `base.png`）。
  2. 使用 **Grounding DINO** 根據文字描述（如 `"image of sky and grass"`）找到對應 bounding box。
  3. 在該 bounding box 中，使用 **CLIPSeg** 根據文字（如 `"grass land"`）預測更細緻的 mask。
  4. 將得到的 mask + 原圖縮放到適合 PowerPaint 的大小，送入  
     `StableDiffusionInpaintPipeline`（`Sanster/PowerPaint-V1-stable-diffusion-inpainting`）。
  5. 產生新圖後再放回原解析度，疊回原圖中 mask 指定位置。
- **輸出**：
  - `debug_groundingdino_box.png`：畫出 Grounding DINO 的 bounding box。
  - `debug_clipseg_mask.png`：CLIPSeg 產生的最終 mask。
  - `final_result.png`：PowerPaint 完成 inpainting 後的成品。


