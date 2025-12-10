import argparse
import os
import json
from PIL import Image, ImageDraw

BASE_DIR = "base_slide"          # 固定看這個資料夾底下的 PNG 檔
BBOX_JSON = "bboxes.json"        # 存 bounding box 資訊的 JSON 檔


def load_bbox_db(path: str):
    """讀取既有 JSON，如果不存在就回傳空 dict。"""
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_bbox_db(path: str, data):
    """把資料寫回 JSON 檔。"""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def visualize_all_bboxes():
    """
    從 bboxes.json 讀出所有頁面的 bbox，
    把每張 base_slide/<ID>.png 上的 bbox 畫成綠色，
    輸出到 ./test/debug_<ID>.png
    """
    db = load_bbox_db(BBOX_JSON)
    if not db:
        print(f"⚠ {BBOX_JSON} 是空的或不存在，沒有任何 bbox 可以顯示。")
        return

    os.makedirs("./test", exist_ok=True)

    print(f"🔍 從 {BBOX_JSON} 中讀出 {len(db)} 個 page 的 bbox 設定")

    for img_id, bbox_dict in db.items():
        image_path = os.path.join(BASE_DIR, f"{img_id}.png")
        if not os.path.exists(image_path):
            print(f"⚠ 跳過 {img_id}：找不到圖片 {image_path}")
            continue

        img = Image.open(image_path).convert("RGB")
        W, H = img.size
        print(f"📄 {img_id}: 讀取圖片 {image_path} ({W}x{H})，共有 {len(bbox_dict)} 個 bbox")

        debug_img = img.copy()
        draw = ImageDraw.Draw(debug_img)

        for k, bbox in bbox_dict.items():
            x1, y1, x2, y2 = bbox
            draw.rectangle([x1, y1, x2, y2], outline="green", width=3)

        out_name = f"debug_{img_id}.png"
        out_path = os.path.join("./test", out_name)
        debug_img.save(out_path)
        print(f"✅ 已輸出：{out_path}")

    print("🎉 全部頁面的 bbox 可視化完成")


def main():
    parser = argparse.ArgumentParser(
        description="在 base_slide 資料夾內指定 PNG 上畫出 bounding box，並輸出 debug 圖，或用 -v 可視化所有已標記的 bbox。"
    )
    parser.add_argument(
        "-v",
        "--visualize-all",
        action="store_true",
        help="只根據 bboxes.json 把所有頁面的 bbox 畫成綠色並輸出，不新增 bbox",
    )

    # 下面這些參數只在「不是 -v 模式」時才會使用
    parser.add_argument(
        "filename",
        type=str,
        nargs="?",
        help="圖片 ID（例如 011，不用帶路徑和副檔名，程式會看 base_slide/011.png）",
    )
    parser.add_argument("x1", type=int, nargs="?", help="bounding box 左上角 x 座標")
    parser.add_argument("y1", type=int, nargs="?", help="bounding box 左上角 y 座標")
    parser.add_argument("x2", type=int, nargs="?", help="bounding box 右下角 x 座標")
    parser.add_argument("y2", type=int, nargs="?", help="bounding box 右下角 y 座標")
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="輸出圖片檔名（預設：在 ./test/ 輸出 debug_<ID>.png）",
    )
    parser.add_argument(
        "-s",
        "--save",
        action="store_true",
        help="同時把這個 bounding box 記錄到 bboxes.json 裡（以 dictionary 方式累積）",
    )

    args = parser.parse_args()

    # ---------- 模式一：-v 可視化全部 bbox ----------
    if args.visualize_all:
        visualize_all_bboxes()
        return

    # ---------- 模式二：標註單一 bbox +（可選）寫入 JSON ----------
    # 檢查必要參數有沒有給
    if args.filename is None or args.x1 is None or args.y1 is None or args.x2 is None or args.y2 is None:
        raise SystemExit("缺少必要參數：需要 filename x1 y1 x2 y2，或改用 -v 模式")

    # 組出實際圖片路徑： base_slide/<ID>.png
    image_path = os.path.join(BASE_DIR, args.filename + ".png")

    if not os.path.exists(image_path):
        raise FileNotFoundError(f"找不到圖片：{image_path}")

    # 讀圖
    img = Image.open(image_path).convert("RGB")
    W, H = img.size
    print(f"讀取圖片: {image_path} ({W}x{H})")

    # 先讀 bboxes.json，把「同一張圖片已經存過的 bbox」畫成綠色
    db = load_bbox_db(BBOX_JSON)
    debug_img = img.copy()
    draw = ImageDraw.Draw(debug_img)

    img_id = args.filename  # 比如 "001"
    if img_id in db:
        print(f"🟩 在 {BBOX_JSON} 中找到 {img_id} 既有的 bbox，共 {len(db[img_id])} 個，畫成綠色")
        for k, bbox in db[img_id].items():
            gx1, gy1, gx2, gy2 = bbox
            draw.rectangle([gx1, gy1, gx2, gy2], outline="green", width=3)

    # 確保座標在圖內（對新輸入的 bbox 做 clamp）
    x1 = max(0, min(W - 1, args.x1))
    y1 = max(0, min(H - 1, args.y1))
    x2 = max(0, min(W - 1, args.x2))
    y2 = max(0, min(H - 1, args.y2))

    if x2 <= x1 or y2 <= y1:
        raise ValueError(
            f"無效的 bbox，請確認座標：({args.x1}, {args.y1}, {args.x2}, {args.y2})"
        )

    print(f"使用 bbox: ({x1}, {y1}) -> ({x2}, {y2})")

    # 把這次指定的新 bbox 畫成紅色
    draw.rectangle([x1, y1, x2, y2], outline="red", width=3)

    # 輸出檔名：預設 ./test/debug_<ID>.png
    os.makedirs("./test", exist_ok=True)
    if args.out:
        out_name = args.out
    else:
        out_name = f"debug_{args.filename}.png"
    out_path = os.path.join("./test", out_name)
    debug_img.save(out_path)
    print(f"✅ 已輸出 debug 圖：{out_path}")

    # 如果有帶 -s，就把 bbox 寫進 JSON
    if args.save:
        print(f"💾 正在把 bbox 寫入 {BBOX_JSON} ...")
        if img_id not in db:
            db[img_id] = {}

        # 找下一個 key（"1", "2", ...）
        existing_keys = db[img_id].keys()
        idx = 1
        while f"{idx}" in existing_keys or idx in existing_keys:
            idx += 1
        key = str(idx)

        db[img_id][key] = [x1, y1, x2, y2]

        save_bbox_db(BBOX_JSON, db)
        print(f"✅ 已在 {BBOX_JSON} 中記錄 {img_id} 的 key={key}, bbox={db[img_id][key]}")


if __name__ == "__main__":
    main()
