#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import glob
import shlex
import subprocess
import sys
from pathlib import Path


def run(cmd: list[str], cwd: Path | None = None) -> None:
    """Run a command, stream output, raise on non-zero exit."""
    print(">>", " ".join(shlex.quote(x) for x in cmd))
    subprocess.run(cmd, cwd=str(cwd) if cwd else None, check=True)


def parse_folders(raw: str) -> list[Path]:
    items = [x.strip() for x in raw.split(",") if x.strip()]
    return [Path(x) for x in items]


def zip_folder(folder: Path, out_dir: Path, overwrite: bool) -> Path:
    folder = folder.resolve()
    if not folder.is_dir():
        raise RuntimeError(f"不是目录: {folder}")

    zip_path = (out_dir / f"{folder.name}.zip").resolve()

    if zip_path.exists():
        if overwrite:
            zip_path.unlink()
        else:
            print(f"[SKIP] zip 已存在: {zip_path}")
            return zip_path

    # 在 folder.parent 下执行，避免把绝对路径写进 zip
    run(["zip", "-r", str(zip_path), folder.name], cwd=folder.parent)
    return zip_path


def remove_old_parts(prefix: str) -> None:
    old = sorted(Path(p) for p in glob.glob(prefix + "*"))
    if old:
        print(f"[CLEAN] 删除旧的 part（{len(old)} 个）")
        for p in old:
            try:
                p.unlink()
            except FileNotFoundError:
                pass


def split_zip(zip_path: Path, part_size: str) -> list[Path]:
    """
    使用系统 split：
      split -b 20G xxx.zip xxx.zip.part
    生成：
      xxx.zip.partaa
      xxx.zip.partab
      ...
    """
    zip_path = zip_path.resolve()
    prefix = str(zip_path) + ".part"

    # 避免混入旧 part
    remove_old_parts(prefix)

    run(["split", "-b", part_size, str(zip_path), prefix])

    parts = sorted(Path(p) for p in glob.glob(prefix + "*"))
    if not parts:
        raise RuntimeError("split 未生成任何 part")
    return parts


def upload_all_parts(dbxcli: Path, parts: list[Path], remote_dir: str | None) -> None:
    for p in parts:
        if remote_dir:
            remote_path = remote_dir.rstrip("/") + "/" + p.name
            run([str(dbxcli), "put", str(p), remote_path])
        else:
            run([str(dbxcli), "put", str(p)])


def main():
    ap = argparse.ArgumentParser(
        description="Zip folders, split with linux 'split', upload each part via dbxcli."
    )
    ap.add_argument(
        "--folders",
        "-f",
        required=True,
        help="逗号分隔的文件夹列表，例如: dir1,dir2,/data/dir3",
    )
    ap.add_argument(
        "--part-size", "-s", default="20G", help="split 的 -b 参数（默认 20G）"
    )
    ap.add_argument(
        "--out-dir", "-o", default=".", help="zip 与 part 输出目录（默认当前目录）"
    )
    ap.add_argument(
        "--dbxcli",
        "-d",
        default="./dbxcli-linux-amd64",
        help="dbxcli-linux-amd64 路径（默认 ./dbxcli-linux-amd64）",
    )
    ap.add_argument(
        "--remote-dir",
        "-r",
        default=None,
        help="Dropbox 远端目录（可选），若提供则上传到 remote_dir/<part_filename>",
    )
    ap.add_argument(
        "--overwrite-zip", "-w", action="store_true", help="若 zip 已存在则覆盖"
    )
    ap.add_argument(
        "--keep-zip", "-k", action="store_true", help="split 后保留 zip（默认删除 zip）"
    )
    ap.add_argument(
        "--keep-parts",
        "-p",
        action="store_true",
        help="保留 part 文件（默认：全部 part 上传成功后统一删除）",
    )

    args = ap.parse_args()

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    dbxcli = "dbxcli-linux-amd64"
    # if not dbxcli.exists():
    #     print(f"[ERROR] dbxcli 不存在: {dbxcli}", file=sys.stderr)
    #     sys.exit(2)

    # 检查 zip / split 是否存在
    for bin_name, install_hint in [
        ("zip", "sudo apt-get install -y zip"),
        ("split", "coreutils（一般系统自带）"),
    ]:
        try:
            subprocess.run(
                [bin_name, "--help"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
            )
        except FileNotFoundError:
            print(
                f"[ERROR] 缺少命令: {bin_name}。安装: {install_hint}", file=sys.stderr
            )
            sys.exit(2)

    folders = parse_folders(args.folders)
    if not folders:
        print("[ERROR] 未解析到任何文件夹", file=sys.stderr)
        sys.exit(2)

    for folder in folders:
        print(f"\n=== 处理文件夹: {folder} ===")
        zip_path = zip_folder(folder, out_dir=out_dir, overwrite=args.overwrite_zip)
        print(f"[OK] zip: {zip_path}")

        parts = split_zip(zip_path, part_size=args.part_size)
        print(f"[OK] split 生成 {len(parts)} 个 part")

        # 上传：任何一个失败会抛异常并退出，此时不会删除 parts
        upload_all_parts(dbxcli, parts, remote_dir=args.remote_dir)
        print("[OK] 所有 part 上传成功")

        # 全部上传成功后统一删除 part
        if not args.keep_parts:
            for p in parts:
                try:
                    p.unlink()
                except FileNotFoundError:
                    pass
            print(f"[CLEAN] 已删除 {len(parts)} 个 part")

        # 是否删除 zip
        if not args.keep_zip:
            try:
                zip_path.unlink()
                print(f"[CLEAN] 删除 zip: {zip_path}")
            except FileNotFoundError:
                pass

    print("\n全部完成")


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] 命令执行失败，退出码 {e.returncode}", file=sys.stderr)
        sys.exit(e.returncode)
    except Exception as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        sys.exit(1)
