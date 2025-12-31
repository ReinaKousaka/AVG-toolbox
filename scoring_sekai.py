#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple


def load_score_map(score_json: Path) -> Dict[str, Any]:
    """
    支持两种输入：
    1) 直接是 { "a.json": 123, "b.json": 45 }
    2) 上一版输出结构 { "file_key_counts": {...}, ... }
    """
    with score_json.open("r", encoding="utf-8") as f:
        obj = json.load(f)
    if (
        isinstance(obj, dict)
        and "file_key_counts" in obj
        and isinstance(obj["file_key_counts"], dict)
    ):
        return obj["file_key_counts"]
    if isinstance(obj, dict):
        return obj
    raise ValueError(
        "score_json must be a dict or contain dict under key 'file_key_counts'"
    )


def parse_dirs(dirs_str: str) -> List[Path]:
    parts = [p.strip() for p in dirs_str.split(",") if p.strip()]
    return [Path(p).expanduser().resolve() for p in parts]


def iter_files(dir_path: Path, recursive: bool) -> List[Path]:
    pattern = "**/*" if recursive else "*"
    return sorted([p for p in dir_path.glob(pattern) if p.is_file()])


def get_lookup_key(file_path: Path, base_dir: Path, mode: str) -> str:
    """
    mode:
      - name: 只用文件名 (xxx.json)
      - rel:  用相对路径 (sub/xxx.json)
      - stem: 用不带后缀的文件名 (xxx)
    """
    if mode == "name":
        return file_path.split("/")[-1]
    if mode == "rel":
        return str(file_path.relative_to(base_dir))
    if mode == "stem":
        return file_path.stem
    raise ValueError(f"Unknown key_mode: {mode}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Traverse one or more folders, look up each filename's score from a JSON map, then sort and save."
    )
    parser.add_argument(
        "--input_dirs",
        "-i",
        type=str,
        required=True,
        help="One or more directories, separated by comma",
    )
    parser.add_argument(
        "--score_json",
        "-s",
        type=str,
        required=True,
        help="JSON file containing score map (e.g. {filename: score} or {'file_key_counts': {...}})",
    )
    parser.add_argument(
        "--output_json",
        "-o",
        type=str,
        required=True,
        help="Output JSON path",
    )
    parser.add_argument(
        "--key_mode",
        "-k",
        type=str,
        default="name",
        choices=["name", "rel", "stem"],
        help="How to construct lookup key for score map (default: name)",
    )
    parser.add_argument(
        "--recursive",
        "-r",
        action="store_true",
        help="Recursively traverse subfolders",
    )
    parser.add_argument(
        "--skip_missing",
        "-m",
        action="store_true",
        help="Skip files that do not exist in score map (default: include with score=None and put at end)",
    )
    parser.add_argument(
        "--require_numeric",
        "-n",
        action="store_true",
        help="Require score to be numeric; otherwise treat as missing",
    )

    args = parser.parse_args()

    dirs = parse_dirs(args.input_dirs)
    for d in dirs:
        if not d.exists() or not d.is_dir():
            print(f"[ERROR] Not a directory: {d}", file=sys.stderr)
            return 2

    score_map = load_score_map(Path(args.score_json).expanduser().resolve())
    out_path = Path(args.output_json).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    records: List[Dict[str, Any]] = []
    missing: List[str] = []

    for d in dirs:
        for fp in iter_files(d, args.recursive):
            fp = str(fp).replace(".mp4", ".json")  # 适应 sekai 输出文件名
            key = get_lookup_key(fp, d, args.key_mode)
            score = score_map.get(key, None)

            if args.require_numeric and (not isinstance(score, (int, float))):
                score = None

            if score is None:
                if args.skip_missing:
                    missing.append(str(fp))
                    continue
                missing.append(str(fp))

            records.append(
                {
                    "file": str(fp),
                    "key": key,
                    "score": score,
                }
            )

    # 排序：score 从高到低；None 放最后；同分按 file 字典序稳定排序
    def sort_key(r: Dict[str, Any]) -> Tuple[int, float, str]:
        s = r["score"]
        if isinstance(s, (int, float)):
            return (0, -float(s), r["file"])
        return (1, float("inf"), r["file"])

    records_sorted = sorted(records, key=sort_key)

    out_obj: Dict[str, Any] = {
        "sorted": records_sorted,
        "summary": {
            "num_dirs": len(dirs),
            "dirs": [str(d) for d in dirs],
            "num_files_total_seen": len(records_sorted)
            + (0 if not args.skip_missing else len(missing)),
            "num_included": len(records_sorted),
            "num_missing_or_invalid_score": len(missing),
            "key_mode": args.key_mode,
            "recursive": bool(args.recursive),
            "skip_missing": bool(args.skip_missing),
            "require_numeric": bool(args.require_numeric),
        },
    }
    if missing:
        out_obj["missing"] = missing

    with out_path.open("w", encoding="utf-8") as f:
        json.dump(out_obj, f, ensure_ascii=False, indent=2)

    print(f"[OK] Wrote sorted list to: {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
