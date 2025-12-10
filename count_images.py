import os
import argparse
from collections import Counter


def main():
    parser = argparse.ArgumentParser(
        description="統計 MagicBrush 轉檔後 images 資料夾裡各 split 的 _source.png 數量，並檢查 mask/target 是否齊全。"
    )
    parser.add_argument(
        "--root",
        type=str,
        default="./magicbrush_converted",
        help="MagicBrush 轉檔根目錄（預設: ./magicbrush_converted）",
    )
    parser.add_argument(
        "--images-dir-name",
        type=str,
        default="images",
        help="圖片子資料夾名稱（預設: images）",
    )
    args = parser.parse_args()

    images_dir = os.path.join(args.root, args.images_dir_name)

    if not os.path.isdir(images_dir):
        raise SystemExit(f"找不到 images 資料夾：{images_dir}")

    counts = Counter()
    incomplete = 0
    total = 0

    for fname in sorted(os.listdir(images_dir)):
        # 只看 *_source.png
        if not fname.endswith("turn1_source.png"):
            continue

        total += 1

        # 取得 split 名稱（檔名前綴的第一段，例如 train_123... -> train）
        split_name = fname.split("_", 1)[0]

        counts[split_name] += 1

        # 檢查這筆對應的 mask/target 是否存在
        base = fname[: -len("turn1_source.png")]  # 去掉尾巴的 "_source.png"
        mask_name = base + "turn1_mask.png"
        tgt_name = base + "turn1_target.png"

        mask_path = os.path.join(images_dir, mask_name)
        tgt_path = os.path.join(images_dir, tgt_name)

        if not (os.path.exists(mask_path) and os.path.exists(tgt_path)):
            incomplete += 1
            print(f"⚠ 不完整樣本: {fname}")
            if not os.path.exists(mask_path):
                print(f"  - 缺少 mask:   {mask_name}")
            if not os.path.exists(tgt_path):
                print(f"  - 缺少 target: {tgt_name}")

    print("\n=== 統計結果 ===")
    for split, n in sorted(counts.items()):
        print(f"{split}: {n} 筆 (_source.png)")

    print(f"總數: {total} 筆")
    print(f"其中不完整樣本（缺 mask 或 target）: {incomplete} 筆")


if __name__ == "__main__":
    main()
