import os
import argparse

def main():
    parser = argparse.ArgumentParser(
        description="把資料夾內所有 .PNG 副檔名改成 .png（只改大小寫）。"
    )
    parser.add_argument(
        "folder",
        type=str,
        help="目標資料夾路徑，例如 base_slide",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="只顯示會怎麼改，不真的重新命名",
    )
    args = parser.parse_args()

    folder = args.folder
    files = os.listdir(folder)
    files.sort()

    any_renamed = False

    for fname in files:
        old_path = os.path.join(folder, fname)

        # 只處理檔案
        if not os.path.isfile(old_path):
            continue

        root, ext = os.path.splitext(fname)

        # 副檔名大小寫不分比較
        if ext.lower() == ".png" and ext != ".png":
            new_fname = root + ".png"
            new_path = os.path.join(folder, new_fname)

            print(f"{fname} -> {new_fname}")
            any_renamed = True

            if not args.dry_run:
                os.rename(old_path, new_path)

    if not any_renamed:
        print("沒有找到需要改名的 .PNG 檔案。")
    elif args.dry_run:
        print("（dry-run 模式，沒有真的重新命名）")

if __name__ == "__main__":
    main()
