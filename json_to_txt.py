import os
import json
import argparse
from typing import Dict, Any, List

from PIL import Image


def load_json(path: str) -> Dict[str, Any]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"找不到 JSON 檔案：{path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def bbox_to_yolo(
    bbox: List[float],
    img_w: int,
    img_h: int,
) -> List[float]:
    """
    將像素座標的 bbox [x1, y1, x2, y2]
    轉成 YOLO 格式的 [x_center_norm, y_center_norm, w_norm, h_norm]。
    """
    x1, y1, x2, y2 = bbox
    # 防呆：確保在圖內
    x1 = max(0, min(img_w - 1, x1))
    x2 = max(0, min(img_w - 1, x2))
    y1 = max(0, min(img_h - 1, y1))
    y2 = max(0, min(img_h - 1, y2))

    w = x2 - x1
    h = y2 - y1
    if w <= 0 or h <= 0:
        raise ValueError(f"無效 bbox：{bbox}")

    x_center = x1 + w / 2.0
    y_center = y1 + h / 2.0

    # 正規化到 0~1
    return [
        x_center / img_w,
        y_center / img_h,
        w / img_w,
        h / img_h,
    ]


def convert_meta_to_yolo(
    meta_path: str,
    images_dir: str,
    labels_dir: str,
    image_key: str = "target",
    class_id: int = 0,
):
    """
    將一個 meta.json 轉成 YOLO .txt 標註檔。

    meta_path : train_meta.json / dev_meta.json
    images_dir: 對應影像所在資料夾，例如 ./complete_slide/train
    labels_dir: 輸出的 YOLO label 資料夾，例如 ./yolo_labels/train
    image_key : 使用哪個欄位當作圖像檔名（預設: 'target'，也可改 'source'）
    class_id  : YOLO 的類別編號（預設 0）
    """
    print(f"🔧 轉換 {meta_path} -> YOLO txt，影像目錄 = {images_dir}")

    data = load_json(meta_path)
    samples = data.get("samples", [])
    if not isinstance(samples, list):
        raise ValueError(f"{meta_path} 結構錯誤，samples 不是 list")

    ensure_dir(labels_dir)

    converted = 0
    skipped_no_image = 0
    skipped_no_bbox = 0

    for i, sample in enumerate(samples):
        img_name = sample.get(image_key, None)
        bbox = sample.get("bbox", None)

        if not img_name:
            print(f"⚠ 第 {i} 筆：缺少 '{image_key}' 欄位，跳過")
            skipped_no_image += 1
            continue

        if not bbox or len(bbox) != 4:
            print(f"⚠ 第 {i} 筆：bbox 無效，跳過；sample = {sample}")
            skipped_no_bbox += 1
            continue

        img_path = os.path.join(images_dir, img_name)
        if not os.path.exists(img_path):
            print(f"⚠ 找不到影像檔 {img_path}，跳過")
            skipped_no_image += 1
            continue

        # 取得影像尺寸
        with Image.open(img_path) as im:
            w, h = im.size

        try:
            x_c, y_c, bw, bh = bbox_to_yolo(bbox, w, h)
        except ValueError as e:
            print(f"⚠ 第 {i} 筆 bbox 轉換失敗：{e}，跳過")
            skipped_no_bbox += 1
            continue

        # label 檔名：跟影像同名但副檔名改成 .txt
        base, _ = os.path.splitext(img_name)
        label_path = os.path.join(labels_dir, base + ".txt")

        # YOLO 格式：class x_center y_center width height
        line = f"{class_id} {x_c:.6f} {y_c:.6f} {bw:.6f} {bh:.6f}\n"

        # 一張圖目前假設只有一個 bbox，直接覆寫/建立檔案
        with open(label_path, "w", encoding="utf-8") as f:
            f.write(line)

        converted += 1

    print(
        f"✅ 完成 {meta_path}: 成功 {converted} 筆，"
        f"缺圖跳過 {skipped_no_image}，bbox 問題跳過 {skipped_no_bbox}"
    )


def main():
    parser = argparse.ArgumentParser(
        description="將 train_meta.json / dev_meta.json 轉成 YOLO txt 標註格式。"
    )
    parser.add_argument(
        "--train-meta",
        type=str,
        default="train_meta.json",
        help="train meta JSON 路徑（預設: train_meta.json）",
    )
    parser.add_argument(
        "--dev-meta",
        type=str,
        default="dev_meta.json",
        help="dev meta JSON 路徑（預設: dev_meta.json）",
    )
    parser.add_argument(
        "--train-images",
        type=str,
        default="./complete_slide/train",
        help="train 影像所在資料夾（預設: ./complete_slide/train）",
    )
    parser.add_argument(
        "--dev-images",
        type=str,
        default="./complete_slide/dev",
        help="dev 影像所在資料夾（預設: ./complete_slide/dev）",
    )
    parser.add_argument(
        "--train-labels",
        type=str,
        default="./yolo_labels/train",
        help="輸出 train YOLO label 的資料夾（預設: ./yolo_labels/train）",
    )
    parser.add_argument(
        "--dev-labels",
        type=str,
        default="./yolo_labels/dev",
        help="輸出 dev YOLO label 的資料夾（預設: ./yolo_labels/dev）",
    )
    parser.add_argument(
        "--image-key",
        type=str,
        default="target",
        help="使用哪個欄位當圖像檔名（預設: target，可改成 source）",
    )
    parser.add_argument(
        "--class-id",
        type=int,
        default=0,
        help="YOLO 類別 id（預設 0）",
    )
    args = parser.parse_args()

    # train
    if args.train_meta and os.path.exists(args.train_meta):
        convert_meta_to_yolo(
            meta_path=args.train_meta,
            images_dir=args.train_images,
            labels_dir=args.train_labels,
            image_key=args.image_key,
            class_id=args.class_id,
        )
    else:
        print(f"⚠ 找不到 train meta：{args.train_meta}，略過")

    # dev
    if args.dev_meta and os.path.exists(args.dev_meta):
        convert_meta_to_yolo(
            meta_path=args.dev_meta,
            images_dir=args.dev_images,
            labels_dir=args.dev_labels,
            image_key=args.image_key,
            class_id=args.class_id,
        )
    else:
        print(f"⚠ 找不到 dev meta：{args.dev_meta}，略過")


if __name__ == "__main__":
    main()
