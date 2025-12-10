import argparse
import os
import re
import json
import random
from typing import Dict, Tuple, List

from PIL import Image

# 路徑設定
BASE_SLIDE_DIR = "base_slide"
MAGIC_IMG_DIR = "./magicbrush_converted/images"
MAGIC_META_DIR = "./magicbrush_converted/meta"

OUT_TRAIN_DIR = "./complete_slide/train"
OUT_DEV_DIR = "./complete_slide/dev"

BBOX_JSON = "bboxes.json"

TRAIN_META_JSON = "train_meta.json"
DEV_META_JSON = "dev_meta.json"


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


def normalize_filter_id(raw: str) -> str:
    m = re.search(r"(\d+)", raw)
    if not m:
        raise ValueError(f"-i/--id 參數必須包含數字，例如 327726 或 train_327726，現在是: {raw}")
    return str(int(m.group(1)))


def parse_id_set(raw: str) -> set:
    """
    把像 "001,002, 003" 這種字串轉成 {"001","002","003"}，
    空字串 -> 空集合。
    """
    if not raw:
        return set()
    return {item.strip() for item in raw.split(",") if item.strip()}


def load_instruction(split_name: str, magic_id: str) -> str:
    """
    從 ./magicbrush_converted/meta/<split>_<id>_turn1.json 讀取 instruction 當作 prompt。
    例如：magicbrush_converted/meta/train_327726_turn1.json
    """
    meta_filename = f"{split_name}_{magic_id}_turn1.json"
    meta_path = os.path.join(MAGIC_META_DIR, meta_filename)
    if not os.path.exists(meta_path):
        print(f"⚠ 找不到 meta 檔案：{meta_path}，此 id 的 prompt 會設為 None")
        return None

    data = load_json(meta_path, {})
    prompt = data.get("instruction", None)
    if prompt is None:
        print(f"⚠ {meta_path} 中沒有 'instruction' 欄位，prompt 會設為 None")
    return prompt


def main():
    parser = argparse.ArgumentParser(
        description="在指定簡報頁的 bbox 中隨機塞入 MagicBrush (turn1) 圖片，生成扁平化的 source/target/mask slide。"
    )
    parser.add_argument(
        "-p",
        "--page",
        required=True,
        type=str,
        help="簡報頁 ID，例如 001（會讀 base_slide/001.png）",
    )
    parser.add_argument(
        "-pe",
        "--photo-exclude",
        type=str,
        default="",
        help="排除某些 base slide page id（逗號分隔），如果 -p 在這裡面就直接跳過不做。",
    )
    parser.add_argument(
        "-n",
        "--num",
        type=int,
        default=1,
        help="每個 MagicBrush id 要隨機生成幾張合成圖（預設 1）",
    )
    parser.add_argument(
        "-i",
        "--id",
        type=str,
        default=None,
        help="只使用指定的 MagicBrush id，例如 327726 或 train_327726；不給則所有 id 都做",
    )
    parser.add_argument(
        "-ie",
        "--id-exclude",
        type=str,
        default="",
        help="排除某些 MagicBrush 圖片 id（逗號分隔，例如：'9,327726'）。",
    )
    args = parser.parse_args()

    page_id = args.page
    num_per_id = args.num
    filter_id_raw = args.id
    page_exclude_set = parse_id_set(args.id_exclude)
    photo_exclude_set = parse_id_set(args.photo_exclude)

    # --- 檢查這個 page 要不要直接被排除 ---
    if page_id in page_exclude_set:
        print(f"🚫 page_id={page_id} 在 -ie 排除列表中，這次不做任何輸出，直接結束。")
        return

    base_path = os.path.join(BASE_SLIDE_DIR, f"{page_id}.png")
    if not os.path.exists(base_path):
        raise FileNotFoundError(f"找不到 base slide 圖片：{base_path}")
    base_img = Image.open(base_path).convert("RGB")
    W, H = base_img.size
    print(f"📄 使用 base slide: {base_path} ({W}x{H})")

    bboxes = load_bboxes_for_page(page_id)
    print(f"📦 在 bboxes.json 中找到 {len(bboxes)} 個 bbox 設置")

    grouped = load_magicbrush_turn1(MAGIC_IMG_DIR)
    print(f"🖼 train split MagicBrush id 數量: {len(grouped['train'])}")
    print(f"🖼 dev   split MagicBrush id 數量: {len(grouped['dev'])}")

    # --- 只使用指定 id（如果有給 -i） ---
    if filter_id_raw is not None:
        norm_id = normalize_filter_id(filter_id_raw)
        for split_name in ["train", "dev"]:
            if norm_id in grouped[split_name]:
                grouped[split_name] = {norm_id: grouped[split_name][norm_id]}
            else:
                grouped[split_name] = {}
        if not grouped["train"] and not grouped["dev"]:
            raise RuntimeError(
                f"在 {MAGIC_IMG_DIR} 中找不到 id={norm_id} 的 train/dev_<id>_turn1_source/target/mask.png"
            )
        print(f"🎯 只使用 id={norm_id}")

    # --- 排除 photo_exclude 中指定的 MagicBrush id ---
    if photo_exclude_set:
        for split_name in ["train", "dev"]:
            before = len(grouped[split_name])
            for ex_id in list(grouped[split_name].keys()):
                if ex_id in photo_exclude_set:
                    grouped[split_name].pop(ex_id, None)
            after = len(grouped[split_name])
            if before != after:
                print(f"🚫 split={split_name}: 根據 -pe 排除了一些 id，剩下 {after} 個 id")

    os.makedirs(OUT_TRAIN_DIR, exist_ok=True)
    os.makedirs(OUT_DEV_DIR, exist_ok=True)

    # meta 結構固定為 { "samples": [ ... ] }
    train_meta = load_json(TRAIN_META_JSON, {"samples": []})
    dev_meta = load_json(DEV_META_JSON, {"samples": []})

    # 🔹 建立「已存在檔名」集合，避免重複（包含 source/target/mask）
    def build_existing_name_set(meta_obj):
        names = set()
        for s in meta_obj.get("samples", []):
            for k in ("source", "target", "mask"):
                if k in s:
                    names.add(s[k])
        return names

    train_existing_names = build_existing_name_set(train_meta)
    dev_existing_names = build_existing_name_set(dev_meta)

    for split_name, out_dir, meta_obj, meta_path, existing_names in [
        ("train", OUT_TRAIN_DIR, train_meta, TRAIN_META_JSON, train_existing_names),
        ("dev", OUT_DEV_DIR, dev_meta, DEV_META_JSON, dev_existing_names),
    ]:
        ids_dict = grouped[split_name]
        if not ids_dict:
            continue

        print(f"➡ 處理 split={split_name}, id 數量={len(ids_dict)}")

        for magic_id in sorted(ids_dict.keys(), key=lambda x: int(x)):
            paths = ids_dict[magic_id]
            print(
                f"  - id={magic_id} ({split_name}), 這個 id 會生成 {num_per_id} 張合成圖"
            )

            # 讀這個 id 對應的 prompt（instruction）
            prompt = load_instruction(split_name, magic_id)

            src_patch = Image.open(paths["source"]).convert("RGBA")
            tgt_patch = Image.open(paths["target"]).convert("RGBA")
            msk_patch = Image.open(paths["mask"]).convert("L")

            for idx in range(1, num_per_id + 1):
                idx_str = f"{idx:03d}"

                bbox = random.choice(bboxes)
                left, top, new_w, new_h = compute_placement(bbox, src_patch.size)
                x1, y1, x2, y2 = left, top, left + new_w, top + new_h

                # <page>_<id>_<source/target/mask>_<n>.png
                src_name = f"{page_id}_{magic_id}_source_{idx_str}.png"
                tgt_name = f"{page_id}_{magic_id}_target_{idx_str}.png"
                mask_name = f"{page_id}_{magic_id}_mask_{idx_str}.png"

                out_source_path = os.path.join(out_dir, src_name)
                out_target_path = os.path.join(out_dir, tgt_name)
                out_mask_path = os.path.join(out_dir, mask_name)

                # 🔸 檢查檔名是否已經出現（meta 裡 or 檔案系統）
                if (
                    src_name in existing_names
                    or tgt_name in existing_names
                    or mask_name in existing_names
                    or os.path.exists(out_source_path)
                    or os.path.exists(out_target_path)
                    or os.path.exists(out_mask_path)
                ):
                    print(
                        f"⚠ 檔名重複，跳過 sample {idx_str}: "
                        f"{src_name}, {tgt_name}, {mask_name}"
                    )
                    continue

                print(
                    f"     🔁 sample {idx_str}: split={split_name}, page={page_id}, id={magic_id}, "
                    f"bbox=({x1},{y1},{x2},{y2})"
                )

                # source slide
                slide_source = paste_patch(base_img, src_patch, left, top, new_w, new_h)
                slide_source.save(out_source_path)

                # target slide
                slide_target = paste_patch(base_img, tgt_patch, left, top, new_w, new_h)
                slide_target.save(out_target_path)

                # mask slide
                base_mask = Image.new("L", (W, H), 0)
                mask_img_res = paste_patch(base_mask, msk_patch, left, top, new_w, new_h)
                mask_img_res.save(out_mask_path)

                # 更新「已存在檔名」集合
                existing_names.add(src_name)
                existing_names.add(tgt_name)
                existing_names.add(mask_name)

                # meta 只存四個檔名 + bbox + prompt
                meta_obj["samples"].append(
                    {
                        "source": src_name,
                        "target": tgt_name,
                        "mask": mask_name,
                        "bbox": [x1, y1, x2, y2],
                        "prompt": prompt,
                    }
                )

        save_json(meta_path, meta_obj)
        print(f"💾 已更新 {meta_path}")

    print("✅ 完成所有合成與標註存檔")
    print(f"  - train images: {OUT_TRAIN_DIR}")
    print(f"  - dev   images: {OUT_DEV_DIR}")
    print(f"  - train meta : {TRAIN_META_JSON}")
    print(f"  - dev   meta : {DEV_META_JSON}")


if __name__ == "__main__":
    main()
