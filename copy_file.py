import os
import shutil
from pathlib import Path


def copy_nested_files(da3_dirs):
    """
    da3_dirs: list[str]  # 例如 ["abc_da3", "def_da3"]
    """

    # 需要匹配的文件后缀
    target_suffixes = ("extrinsics_da3nested.npy", "intrinsics_da3nested.npy")

    for da3_dir in da3_dirs:
        da3_path = Path(da3_dir)
        if not da3_path.is_dir():
            print(f"跳过不存在的目录: {da3_path}")
            continue

        # 构造对应的 frustum 目录
        frustum_dir = Path(str(da3_path).replace("_da3", "_frustum"))

        if not frustum_dir.exists():
            print(f"目标 frustum 目录不存在，跳过: {frustum_dir}")
            continue

        # 遍历 _da3 目录下所有文件
        for file in da3_path.rglob("*"):
            if file.is_file() and file.name.endswith(target_suffixes):
                # 构造目标路径
                target_file = frustum_dir / file.name

                # 执行复制
                shutil.copy2(file, target_file)
                print(f"已复制: {file}  ->  {target_file}")


# 示例使用
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Copy specific nested files from _da3 to _frustum directories."
    )
    parser.add_argument(
        "input_dirs",
        type=str,
        help="seperate by commas",
    )
    args = parser.parse_args()
    da3_folders = args.input_dirs.split(",")
    copy_nested_files(da3_folders)
