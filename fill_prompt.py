import os
import re
import json
from typing import Dict, Any, Tuple

MAGIC_META_DIR = "magicbrush_converted/meta"
TRAIN_META_JSON = "train_meta.json"
DEV_META_JSON = "dev_meta.json"


def load_json(path: str, default):
    if not os.path.exists(path):
        return default
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: str, data: Dict[str, Any]):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def parse_magic_id_from_source(source_name: str) -> Tuple[str, str, str]:
    """
    從 source 檔名解析出 (page_id, magic_id, idx)

    檔名格式假設為：
      <page>_<magic_id>_source_<idx>.png

    回傳: (page_id, magic_id, idx)

    例如：
      '001_327726_source_001.png'
      -> ('001', '327726', '001')
    """
    m = re.match(r"^(\d+)_([0-9]+)_source_(\d+)\.png$", source_name)
    if not m:
        raise ValueError(f"無法從 source 檔名解析出 magic_id: {source_name}")
    return m.group(1), m.group(2), m.group(3)


def load_instruction(split_name: str, magic_id: str) -> str:
    """
    從 ./magicbrush_converted/meta/<split>_<id>_turn1.json 讀取 instruction 當作 prompt。
    例如：magicbrush_converted/meta/train_327726_turn1.json
    """
    meta_filename = f"{split_name}_{magic_id}_turn1.json"
    meta_path = os.path.join(MAGIC_META_DIR, meta_filename)

    if not os.path.exists(meta_path):
        print(f"⚠ 找不到 meta 檔案：{meta_path}，此 sample 的 prompt 無法補上")
        return None

    data = load_json(meta_path, {})
    prompt = data.get("instruction", None)
    if prompt is None:
        print(f"⚠ {meta_path} 中沒有 'instruction' 欄位，此 sample 的 prompt 會維持 None")
    return prompt


def backfill_meta(meta_path: str, split_name: str):
    """
    補齊指定 meta 檔案中的 prompt 欄位。

    split_name: "train" 或 "dev"
    """
    meta = load_json(meta_path, None)
    if meta is None:
        print(f"⚠ 找不到 {meta_path}，跳過")
        return

    samples = meta.get("samples", [])
    if not isinstance(samples, list):
        print(f"⚠ {meta_path} 結構異常，'samples' 不是 list，跳過")
        return

    total = len(samples)
    already_has = 0
    filled = 0
    failed = 0

    # cache: (split, magic_id) -> prompt，避免同一 id 重複讀檔
    prompt_cache: Dict[Tuple[str, str], str] = {}

    print(f"\n🔧 處理 {meta_path}（split={split_name}），samples 數量: {total}")

    for i, sample in enumerate(samples):
        # 如果已經有 prompt，而且非空字串，就直接跳過
        if "prompt" in sample and sample["prompt"]:
            already_has += 1
            continue

        source_name = sample.get("source", None)
        if not source_name:
            print(f"⚠ 第 {i} 筆 sample 沒有 'source' 欄位，無法解析 magic_id，跳過")
            failed += 1
            continue

        try:
            page_id, magic_id, idx = parse_magic_id_from_source(source_name)
        except ValueError as e:
            print(f"⚠ 第 {i} 筆 sample：{e}，跳過")
            failed += 1
            continue

        cache_key = (split_name, magic_id)
        if cache_key in prompt_cache:
            prompt = prompt_cache[cache_key]
        else:
            prompt = load_instruction(split_name, magic_id)
            prompt_cache[cache_key] = prompt

        if prompt is None:
            # 找不到 instruction，就保持 prompt 無/None
            failed += 1
            continue

        # 補上 prompt
        sample["prompt"] = prompt
        filled += 1

    # 寫回檔案
    save_json(meta_path, meta)

    print(
        f"✅ 完成 {meta_path}：總共 {total} 筆，"
        f"{already_has} 筆原本就有 prompt，"
        f"{filled} 筆成功補上，{failed} 筆失敗（找不到對應 meta 或 instruction）。"
    )


def main():
    # train_meta
    backfill_meta(TRAIN_META_JSON, "train")

    # dev_meta
    backfill_meta(DEV_META_JSON, "dev")


if __name__ == "__main__":
    main()
