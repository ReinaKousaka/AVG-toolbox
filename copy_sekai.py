import os
import shutil

# ================== 参数配置 ==================
TXT_PATH = "sekai-good.txt"  # 上一步生成的 txt
SRC_DIR = "sekai-game-walking"  # 原始 mp4 / npz 文件夹
DST_DIR = "sekai-game-walking-good"  # 目标文件夹
# =============================================


def read_base_names(txt_path):
    """读取 txt 中的基础文件名"""
    with open(txt_path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def copy_files(base_names, src_dir, dst_dir):
    """拷贝 mp4 和 npz 文件"""
    os.makedirs(dst_dir, exist_ok=True)

    missing_files = []

    for name in base_names:
        for ext in (".mp4", ".npz"):
            src_path = os.path.join(src_dir, name + ext)
            dst_path = os.path.join(dst_dir, name + ext)

            if os.path.isfile(src_path):
                shutil.copy2(src_path, dst_path)
            else:
                missing_files.append(src_path)

    return missing_files


def main():
    base_names = read_base_names(TXT_PATH)
    missing = copy_files(base_names, SRC_DIR, DST_DIR)

    print(f"完成拷贝，共处理 {len(base_names)} 个文件名")

    if missing:
        print("以下文件未找到：")
        for f in missing:
            print("  ", f)


if __name__ == "__main__":
    main()
