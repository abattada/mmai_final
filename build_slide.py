import argparse
import os
import re
import json
import random
from typing import Dict, Tuple, List
from concurrent.futures import ThreadPoolExecutor, as_completed

from PIL import Image

# 路徑設定
BASE_SLIDE_DIR = "base_slide"
MAGIC_IMG_DIR = "./magicbrush_converted/images"
MAGIC_META_DIR = "./magicbrush_converted/meta"

DATASET_ROOT = "./dataset"  # 輸出影像：dataset/<train|validation|test>
GT_ROOT = "./gt"            # 輸出標註：gt/<train|validation|test>/meta.json

BBOX_JSON = "bboxes.json"

# MagicBrush train split：前 4000 張當 train，剩下尾端 512 張當 validation
TRAIN_MAX_IDS = 4000
VAL_TAIL_IDS = 512


def load_json(path: str, default):
    if not os.path.exists(path):
        return default
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: str, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def load_magicbrush_turn1(image_dir: str) -> Dict[str, Dict[str, Dict[str, str]]]:
    """
    掃描 ./magicbrush_converted/images 底下所有
    train_<id>_turn1_source.png
    dev_<id>_turn1_source.png

    回傳：
    {
      "train": {
        "<id>": {"source":..., "target":..., "mask":...},
        ...
      },
      "dev": {
        "<id>": {...},
        ...
      }
    }
    """
    grouped: Dict[str, Dict[str, Dict[str, str]]] = {
        "train": {},
        "dev": {},
    }

    pattern = re.compile(r"^(train|dev)_(\d+)_turn1_source\.png$")

    for fname in os.listdir(image_dir):
        m = pattern.match(fname)
        if not m:
            continue

        split, magic_id = m.group(1), m.group(2)
        base = f"{split}_{magic_id}_turn1"

        src_path = os.path.join(image_dir, f"{base}_source.png")
        tgt_path = os.path.join(image_dir, f"{base}_target.png")
        msk_path = os.path.join(image_dir, f"{base}_mask.png")

        if not (os.path.exists(src_path) and os.path.exists(tgt_path) and os.path.exists(msk_path)):
            continue

        grouped[split][magic_id] = {
            "source": src_path,
            "target": tgt_path,
            "mask": msk_path,
        }

    if not grouped["train"] and not grouped["dev"]:
        raise RuntimeError(
            f"在 {image_dir} 找不到任何 train/dev_<id>_turn1_source/target/mask.png triplet"
        )

    return grouped


def load_bboxes_for_page(page_id: str) -> List[Tuple[int, int, int, int]]:
    db = load_json(BBOX_JSON, {})
    if page_id not in db:
        raise KeyError(f"{BBOX_JSON} 中沒有 key '{page_id}' 的 bbox 設置")

    bdict = db[page_id]
    boxes = []
    for k in sorted(bdict.keys()):
        x1, y1, x2, y2 = bdict[k]
        boxes.append((int(x1), int(y1), int(x2), int(y2)))
    if not boxes:
        raise RuntimeError(f"{BBOX_JSON} 中 '{page_id}' 雖然有 key，但沒有任何 bbox 設置")
    return boxes


def compute_placement(
    bbox: Tuple[int, int, int, int],
    patch_size: Tuple[int, int],
) -> Tuple[int, int, int, int]:
    bx1, by1, bx2, by2 = bbox
    bw = bx2 - bx1
    bh = by2 - by1

    pw, ph = patch_size
    if pw <= 0 or ph <= 0:
        raise ValueError("patch 尺寸不合法")

    base_scale = min(bw / pw, bh / ph, 1.0)
    new_w0 = max(1, int(pw * base_scale))
    new_h0 = max(1, int(ph * base_scale))

    new_w, new_h = new_w0, new_h0

    MIN_SIDE = 200
    if new_w0 >= MIN_SIDE and new_h0 >= MIN_SIDE:
        min_scale = max(MIN_SIDE / new_w0, MIN_SIDE / new_h0)
        min_scale = min(max(min_scale, 0.0), 1.0)
        s = random.uniform(min_scale, 1.0)
        new_w = max(1, int(new_w0 * s))
        new_h = max(1, int(new_h0 * s))
        new_w = max(new_w, MIN_SIDE)
        new_h = max(new_h, MIN_SIDE)
        new_w = min(new_w, bw)
        new_h = min(new_h, bh)

    max_x = bx2 - new_w
    max_y = by2 - new_h
    if max_x < bx1 or max_y < by1:
        left = bx1
        top = by1
    else:
        left = random.randint(bx1, max_x)
        top = random.randint(by1, max_y)

    return left, top, new_w, new_h


def paste_patch(
    base_img: Image.Image,
    patch_img: Image.Image,
    left: int,
    top: int,
    new_w: int,
    new_h: int,
) -> Image.Image:
    patch_resized = patch_img.resize((new_w, new_h), resample=Image.LANCZOS)
    out = base_img.copy()
    if patch_resized.mode == "RGBA":
        out.paste(patch_resized, (left, top), mask=patch_resized)
    elif patch_resized.mode == "L" and base_img.mode == "L":
        out.paste(patch_resized, (left, top))
    else:
        out.paste(patch_resized, (left, top))
    return out


def parse_page_list(raw: str) -> List[str]:
    """
    "001, 002,003" -> ["001","002","003"]
    """
    if not raw:
        return []
    return [item.strip() for item in raw.split(",") if item.strip()]


def load_instruction(magic_split: str, magic_id: str) -> str:
    """
    從 ./magicbrush_converted/meta/<magic_split>_<id>_turn1.json 讀取 instruction 當作 prompt。
    magic_split: "train" 或 "dev"
    """
    meta_filename = f"{magic_split}_{magic_id}_turn1.json"
    meta_path = os.path.join(MAGIC_META_DIR, meta_filename)
    if not os.path.exists(meta_path):
        print(f"⚠ 找不到 meta 檔案：{meta_path}，此 id 的 prompt 會設為 None")
        return None

    data = load_json(meta_path, {})
    prompt = data.get("instruction", None)
    if prompt is None:
        print(f"⚠ {meta_path} 中沒有 'instruction' 欄位，prompt 會設為 None")
    return prompt


def build_existing_name_set(meta_obj):
    """
    從既有 meta 中蒐集所有出現過的檔名（source/target/mask），避免重複。
    """
    names = set()
    for s in meta_obj.get("samples", []):
        for k in ("source", "target", "mask"):
            if k in s:
                names.add(s[k])
    return names


def worker_task(
    split_name: str,
    page_id: str,
    magic_id: str,
    idx_str: str,
    magic_split: str,
    id_info: Dict[str, str],
    out_dir: str,
) -> dict | None:
    """
    單一 sample 的工作：
    - 讀取 base slide / MagicBrush patches
    - 隨機選 bbox 與放置位置
    - 產生 source/target/mask 圖片
    - 回傳 meta entry（或 None 表示跳過）
    """
    base_path = os.path.join(BASE_SLIDE_DIR, f"{page_id}.png")
    if not os.path.exists(base_path):
        print(f"⚠ [thread:{split_name}] 找不到 base slide 圖片：{base_path}，跳過。")
        return None

    try:
        bboxes = load_bboxes_for_page(page_id)
    except Exception as e:
        print(f"⚠ [thread:{split_name}] 載入 {page_id} 的 bbox 失敗：{e}，跳過。")
        return None

    if not bboxes:
        print(f"⚠ [thread:{split_name}] {page_id} 沒有任何 bbox，跳過。")
        return None

    base_img = Image.open(base_path).convert("RGB")
    W, H = base_img.size

    src_path = id_info["source"]
    tgt_path = id_info["target"]
    msk_path = id_info["mask"]
    prompt = id_info["prompt"]

    if not (os.path.exists(src_path) and os.path.exists(tgt_path) and os.path.exists(msk_path)):
        print(f"⚠ [thread:{split_name}] MagicBrush 圖片缺失 id={magic_id}，跳過。")
        return None

    src_patch = Image.open(src_path).convert("RGBA")
    tgt_patch = Image.open(tgt_path).convert("RGBA")
    msk_patch = Image.open(msk_path).convert("L")

    bbox = random.choice(bboxes)
    left, top, new_w, new_h = compute_placement(bbox, src_patch.size)
    x1, y1, x2, y2 = left, top, left + new_w, top + new_h

    src_name = f"{page_id}_{magic_id}_source_{idx_str}.png"
    tgt_name = f"{page_id}_{magic_id}_target_{idx_str}.png"
    mask_name = f"{page_id}_{magic_id}_mask_{idx_str}.png"

    out_source_path = os.path.join(out_dir, src_name)
    out_target_path = os.path.join(out_dir, tgt_name)
    out_mask_path = os.path.join(out_dir, mask_name)

    # 實際輸出圖片
    slide_source = paste_patch(base_img, src_patch, left, top, new_w, new_h)
    slide_source.save(out_source_path)

    slide_target = paste_patch(base_img, tgt_patch, left, top, new_w, new_h)
    slide_target.save(out_target_path)

    base_mask = Image.new("L", (W, H), 0)
    mask_img_res = paste_patch(base_mask, msk_patch, left, top, new_w, new_h)
    mask_img_res.save(out_mask_path)

    print(
        f"[{split_name}] page={page_id}, magic_id={magic_id}, sample={idx_str}, "
        f"bbox=({x1},{y1},{x2},{y2})"
    )

    return {
        "source": src_name,
        "target": tgt_name,
        "mask": mask_name,
        "bbox": [x1, y1, x2, y2],
        "prompt": prompt,
    }


def main():
    parser = argparse.ArgumentParser(
        description=(
            "根據 bboxes.json 與 MagicBrush (turn1)，"
            "產生 dataset/<train|validation|test> 與 gt/<train|validation|test>/meta.json。"
        )
    )
    parser.add_argument(
        "-n",
        "--num",
        type=int,
        default=1,
        help="每個 MagicBrush 圖片在每張背景投影片上要隨機生成幾張合成圖（預設 1）",
    )
    parser.add_argument(
        "-t",
        "--test-run",
        action="store_true",
        help="測試模式：每個分區只產生一張照片（1 個 page × 1 個 MagicBrush id × 1 張 sample）",
    )
    parser.add_argument(
        "-w",
        "--workers",
        type=int,
        default=4,
        help="平行執行的 thread 數量（預設 4，設為 1 則改為單線程執行）",
    )
    args = parser.parse_args()

    # 背景投影片分配到不同 split
    train_pages = [
        "001", "002", "003", "004", "005", "006", "008", "010",
        "013", "014", "017", "023", "025", "027", "029", "030",
        "031", "032", "034", "037", "039", "042", "043", "045",
        "048", "050", "056", "060", "061", "063", "064", "068", "072"
    ]
    val_pages = [
        "009", "012", "016", "019", "024", "028", "036",
        "040", "053", "054", "059", "065", "069", "073"
    ]
    test_pages = [
        "007", "011", "015", "018", "020", "021", "022", "026",
        "033", "035", "038", "041", "044", "046", "047", "049",
        "051", "052", "055", "057", "058", "062", "066", "067",
        "070", "071", "074", "075"
    ]
    num_per_id = args.num

    if not train_pages and not val_pages and not test_pages:
        raise ValueError("train-pages / val-pages / test-pages 至少要指定一個非空。")

    # 1. 載入 MagicBrush 圖片（turn1）
    grouped_all = load_magicbrush_turn1(MAGIC_IMG_DIR)

    # 2. 依照規則拆成三組：
    #    - train：train split 的前 4000 個 id
    #    - validation：train split 剩下的尾端 512 個 id
    #    - test：dev split 的所有 id
    train_ids_sorted = sorted(grouped_all["train"].keys(), key=lambda x: int(x))
    dev_ids_sorted = sorted(grouped_all["dev"].keys(), key=lambda x: int(x))

    train_ids_for_train = train_ids_sorted[:TRAIN_MAX_IDS]
    remaining_train_ids = train_ids_sorted[TRAIN_MAX_IDS:]
    if len(remaining_train_ids) >= VAL_TAIL_IDS:
        train_ids_for_val = remaining_train_ids[-VAL_TAIL_IDS:]
    else:
        train_ids_for_val = remaining_train_ids

    print(f"🧩 MagicBrush train 總共 {len(train_ids_sorted)} 個 id")
    print(f"  - train 使用前 {len(train_ids_for_train)} 個 id")
    print(f"  - validation 使用後 {len(train_ids_for_val)} 個 id（從剩餘的 train 中取尾端）")
    print(f"🧩 MagicBrush dev 總共 {len(dev_ids_sorted)} 個 id（全部給 test 使用）")

    ids_train_dict = {mid: grouped_all["train"][mid] for mid in train_ids_for_train}
    ids_val_dict = {mid: grouped_all["train"][mid] for mid in train_ids_for_val}
    ids_test_dict = {mid: grouped_all["dev"][mid] for mid in dev_ids_sorted}

    # 3. 準備三個 split 的設定
    splits = {
        "train": {
            "pages": train_pages,
            "magic_split": "train",
            "ids_dict": ids_train_dict,
        },
        "validation": {
            "pages": val_pages,
            "magic_split": "train",
            "ids_dict": ids_val_dict,
        },
        "test": {
            "pages": test_pages,
            "magic_split": "dev",
            "ids_dict": ids_test_dict,
        },
    }

    # 如果是測試模式：每個 split 只拿 1 個 page + 1 個 magic_id，且每個 id 只產生 1 張
    if args.test_run:
        print("⚙ 啟用測試模式：每個 split 只產生一張照片")
        for split_name, cfg in splits.items():
            pages = cfg["pages"]
            ids_dict = cfg["ids_dict"]

            if pages:
                cfg["pages"] = pages[:1]
            if ids_dict:
                first_id = sorted(ids_dict.keys(), key=lambda x: int(x))[0]
                cfg["ids_dict"] = {first_id: ids_dict[first_id]}

        num_per_id = 1  # 測試模式固定每個 id 只生一張

    # 4. 針對 train / validation / test 各別產出：
    #    - dataset/<split>/*.png
    #    - gt/<split>/meta.json
    for split_name, cfg in splits.items():
        pages = cfg["pages"]
        ids_dict = cfg["ids_dict"]
        magic_split = cfg["magic_split"]

        if not pages:
            print(f"⚠ split={split_name} 沒有指定任何背景投影片，跳過。")
            continue
        if not ids_dict:
            print(f"⚠ split={split_name} 沒有任何 MagicBrush 圖片可用，跳過。")
            continue

        out_dir = os.path.join(DATASET_ROOT, split_name)
        gt_dir = os.path.join(GT_ROOT, split_name)
        os.makedirs(out_dir, exist_ok=True)
        os.makedirs(gt_dir, exist_ok=True)

        meta_path = os.path.join(gt_dir, "meta.json")
        meta_obj = load_json(meta_path, {"samples": []})
        existing_names = build_existing_name_set(meta_obj)

        print(f"\n========== 處理 split={split_name} ==========")
        print(f"📄 背景投影片頁數：{len(pages)}")
        print(f"🖼 MagicBrush id 數量：{len(ids_dict)}")
        print(f"📁 輸出影像資料夾：{out_dir}")
        print(f"📁 輸出 GT 檔案：{meta_path}")

        # 預先載入每個 MagicBrush id 的 prompt
        id_infos: Dict[str, Dict[str, str]] = {}
        for magic_id, paths in ids_dict.items():
            prompt = load_instruction(magic_split, magic_id)
            id_infos[magic_id] = {
                "source": paths["source"],
                "target": paths["target"],
                "mask": paths["mask"],
                "prompt": prompt,
            }

        # 建立所有要處理的任務列表
        tasks: List[Tuple[str, str, str, str, Dict[str, str], str]] = []
        for page_id in pages:
            for magic_id in sorted(id_infos.keys(), key=lambda x: int(x)):
                for idx in range(1, num_per_id + 1):
                    idx_str = f"{idx:03d}"
                    src_name = f"{page_id}_{magic_id}_source_{idx_str}.png"
                    tgt_name = f"{page_id}_{magic_id}_target_{idx_str}.png"
                    mask_name = f"{page_id}_{magic_id}_mask_{idx_str}.png"

                    out_source_path = os.path.join(out_dir, src_name)
                    out_target_path = os.path.join(out_dir, tgt_name)
                    out_mask_path = os.path.join(out_dir, mask_name)

                    # 若檔名已出現在 meta 或實體檔案已存在，就直接跳過這個任務
                    if (
                        src_name in existing_names
                        or tgt_name in existing_names
                        or mask_name in existing_names
                        or os.path.exists(out_source_path)
                        or os.path.exists(out_target_path)
                        or os.path.exists(out_mask_path)
                    ):
                        print(
                            f"⚠ [{split_name}] 檔名已存在，略過既有 sample: "
                            f"{src_name}, {tgt_name}, {mask_name}"
                        )
                        continue

                    tasks.append(
                        (
                            split_name,
                            page_id,
                            magic_id,
                            idx_str,
                            magic_split,
                            id_infos[magic_id],
                            out_dir,
                        )
                    )

        print(f"🧵 split={split_name} 總任務數量：{len(tasks)}（threads={args.workers}）")

        # 執行任務（平行或單線程）
        new_samples: List[dict] = []

        if args.workers == 1:
            # 單線程執行，方便 debug
            for t in tasks:
                result = worker_task(*t)
                if result is not None:
                    new_samples.append(result)
                    existing_names.add(result["source"])
                    existing_names.add(result["target"])
                    existing_names.add(result["mask"])
        else:
            with ThreadPoolExecutor(max_workers=args.workers) as executor:
                future_to_task = {
                    executor.submit(worker_task, *t): t for t in tasks
                }
                for future in as_completed(future_to_task):
                    result = future.result()
                    if result is not None:
                        new_samples.append(result)
                        existing_names.add(result["source"])
                        existing_names.add(result["target"])
                        existing_names.add(result["mask"])

        # 把新產生的 samples 加進 meta
        meta_obj["samples"].extend(new_samples)
        save_json(meta_path, meta_obj)
        print(f"💾 split={split_name} 完成，新增 {len(new_samples)} 筆樣本，已更新 {meta_path}")

    print("\n✅ 完成所有 split 的合成與標註輸出")
    print(f"  - 影像：{DATASET_ROOT}/<train|validation|test>")
    print(f"  - GT  ：{GT_ROOT}/<train|validation|test>/meta.json")


if __name__ == "__main__":
    main()
