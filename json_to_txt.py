import os
import json

GT_ROOT = "./gt"          # 裡面有 train/validation/test/meta.json
SPLITS = ["train", "validation", "test"]

# 投影片尺寸
IMG_W = 1280.0
IMG_H = 720.0


def load_json(path, default=None):
    if default is None:
        default = {}
    if not os.path.exists(path):
        return default
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def bbox_to_yolo(x1, y1, x2, y2):
    """把絕對座標轉成 YOLO normalized 格式 (cx, cy, w, h)。"""
    cx = (x1 + x2) / 2.0 / IMG_W
    cy = (y1 + y2) / 2.0 / IMG_H
    w = (x2 - x1) / IMG_W
    h = (y2 - y1) / IMG_H
    return cx, cy, w, h


def main():
    for split in SPLITS:
        meta_path = os.path.join(GT_ROOT, split, "meta.json")
        if not os.path.exists(meta_path):
            print(f"⚠ 找不到 {meta_path}，略過 {split}")
            continue

        data = load_json(meta_path, {"samples": []})
        samples = data.get("samples", [])

        if not samples:
            print(f"⚠ {meta_path} 裡沒有 samples，略過 {split}")
            continue

        # yolo label 檔放在 gt/<split>/yolo/
        yolo_dir = os.path.join(GT_ROOT, split, "label")
        os.makedirs(yolo_dir, exist_ok=True)

        count = 0
        for s in samples:
            x1, y1, x2, y2 = s["bbox"]
            cx, cy, w, h = bbox_to_yolo(x1, y1, x2, y2)
            class_id = 0  # 單一類別

            # 針對 source / target / mask 三種圖各自輸出一個 txt
            for key in ("source", "target", "mask"):
                if key not in s:
                    continue
                img_name = s[key]
                base, _ = os.path.splitext(img_name)
                txt_name = base + ".txt"
                txt_path = os.path.join(yolo_dir, txt_name)

                with open(txt_path, "w", encoding="utf-8") as f:
                    f.write(f"{class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")

                count += 1

        print(f"✅ split={split} 已在 {yolo_dir} 產生 {count} 個 YOLO txt 標註檔（含 source/target/mask）")


if __name__ == "__main__":
    main()
