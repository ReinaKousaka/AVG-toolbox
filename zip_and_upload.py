#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import subprocess
import shlex
import sys
from pathlib import Path
import glob


def run(cmd, cwd=None):
    print(">>", " ".join(shlex.quote(c) for c in cmd))
    subprocess.run(cmd, cwd=cwd, check=True)


def zip_folder(folder: Path, out_dir: Path, overwrite: bool) -> Path:
    folder = folder.resolve()
    if not folder.is_dir():
        raise RuntimeError(f"不是目录: {folder}")

    zip_path = out_dir / f"{folder.name}.zip"

    if zip_path.exists():
        if overwrite:
            zip_path.unlink()
        else:
            print(f"[SKIP] zip 已存在: {zip_path}")
            return zip_path

    # 在父目录执行 zip，避免绝对路径
    run(["zip", "-r", str(zip_path), folder.name], cwd=str(folder.parent))
    return zip_path


def split_zip(zip_path: Path, part_size: str):
    """
    使用系统 split：
    split -b 20G xxx.zip xxx.zip.part
    生成：
    xxx.zip.partaa
    xxx.zip.partab
    ...
    """
    prefix = f"{zip_path}.part"
    run(["split", "-b", part_size, str(zip_path), prefix])

    parts = sorted(Path(p) for p in glob.glob(f"{prefix}*"))
    if not parts:
        raise RuntimeError("split 未生成任何 part")
    return parts


def upload_parts(dbxcli: Path, parts, remote_dir=None):
    for p in parts:
        if remote_dir:
            remote_path = remote_dir.rstrip("/") + "/" + p.name
            run([str(dbxcli), "put", str(p), remote_path])
        else:
            run([str(dbxcli), "put", str(p)])


def parse_folders(s: str):
    return [x.strip() for x in s.split(",") if x.strip()]


def main():
    ap = argparse.ArgumentParser(
        description="Zip folders, split with linux split, upload via dbxcli"
    )
    ap.add_argument(
        "--folders",
        required=True,
        help="逗号分隔的文件夹列表，如: dir1,dir2,/data/dir3",
    )
    ap.add_argument("--part-size", default="20G", help="split 的大小参数（默认 20G）")
    ap.add_argument("--out-dir", default=".", help="zip 与 part 输出目录")
    ap.add_argument(
        "--dbxcli", default="dbxcli-linux-amd64", help="dbxcli-linux-amd64 路径"
    )
    ap.add_argument("--remote-dir", default=None, help="Dropbox 远端目录（可选）")
    ap.add_argument("--overwrite-zip", action="store_true", help="若 zip 已存在则覆盖")
    ap.add_argument(
        "--keep-zip", action="store_true", help="split 后保留 zip（默认删除）"
    )

    args = ap.parse_args()

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    dbxcli = args.dbxcli
    # if not dbxcli.exists():
    #     print(f"dbxcli 不存在: {dbxcli}", file=sys.stderr)
    #     sys.exit(1)

    folders = parse_folders(args.folders)
    if not folders:
        print("未解析到任何文件夹", file=sys.stderr)
        sys.exit(1)

    for f in folders:
        folder = Path(f)
        print(f"\n=== 处理文件夹: {folder} ===")

        zip_path = zip_folder(folder, out_dir, args.overwrite_zip)
        print(f"[OK] zip: {zip_path}")

        parts = split_zip(zip_path, args.part_size)
        print(f"[OK] split 生成 {len(parts)} 个 part")

        upload_parts(dbxcli, parts, args.remote_dir)

        if not args.keep_zip:
            zip_path.unlink()
            print(f"[CLEAN] 删除 zip: {zip_path}")

    print("\n全部完成")


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as e:
        print(f"命令执行失败，退出码 {e.returncode}", file=sys.stderr)
        sys.exit(e.returncode)
    except Exception as e:
        print(f"错误: {e}", file=sys.stderr)
        sys.exit(1)
