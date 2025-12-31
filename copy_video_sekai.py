#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List


def load_sorted_items(sorted_json: Path) -> List[Dict[str, Any]]:
    with sorted_json.open("r", encoding="utf-8") as f:
        obj = json.load(f)
    if (
        not isinstance(obj, dict)
        or "sorted" not in obj
        or not isinstance(obj["sorted"], list)
    ):
        raise ValueError("Input JSON must contain top-level key 'sorted' as a list")
    return obj["sorted"]


def is_num(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def safe_copy(src: Path, dst: Path, overwrite: bool) -> str:
    if dst.exists():
        if not overwrite:
            return "exists_skip"
        # overwrite
        dst.unlink()
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return "copied"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Copy top-N scored videos (json->mp4) from sorted JSON into a new folder."
    )
    parser.add_argument(
        "--sorted_json",
        "-s",
        type=str,
        required=True,
        help="Input JSON containing 'sorted' list",
    )
    parser.add_argument(
        "--output_dir",
        "-o",
        type=str,
        required=True,
        help="Destination folder to copy mp4 files into",
    )
    parser.add_argument(
        "--top_n", "-n", type=int, default=100, help="Copy top N (<=0 means all)"
    )
    parser.add_argument(
        "--start_idx",
        "-t",
        type=int,
        default=0,
        help="Start index within sorted list (default 0)",
    )
    parser.add_argument(
        "--overwrite",
        "-w",
        action="store_true",
        help="Overwrite destination files if exist",
    )
    parser.add_argument(
        "--flat",
        "-f",
        action="store_true",
        help="Flat output (default): copy into output_dir/<filename>.mp4; "
        "If not set, preserve relative path from filesystem root is NOT recommended, so we preserve parent folder name instead.",
    )
    parser.add_argument(
        "--preserve_parent",
        "-p",
        action="store_true",
        help="When not --flat, put into output_dir/<parent_folder>/<filename>.mp4 (helps avoid name collisions)",
    )
    args = parser.parse_args()

    sorted_json = Path(args.sorted_json).expanduser().resolve()
    out_dir = Path(args.output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    items = load_sorted_items(sorted_json)

    # 默认 items 已经按分数从高到低
    filtered: List[Dict[str, Any]] = []
    for it in items:
        fp = it.get("file", None)
        sc = it.get("score", None)
        if not isinstance(fp, str):
            continue
        if not is_num(sc):
            continue
        filtered.append(it)

    sliced = filtered[args.start_idx :]
    if args.top_n > 0:
        sliced = sliced[: args.top_n]

    report = {
        "meta": {
            "sorted_json": str(sorted_json),
            "output_dir": str(out_dir),
            "top_n": args.top_n,
            "start_idx": args.start_idx,
            "overwrite": bool(args.overwrite),
            "flat": bool(args.flat),
            "preserve_parent": bool(args.preserve_parent),
        },
        "copied": [],
        "missing": [],
        "skipped": [],
    }

    for rank, it in enumerate(sliced):
        json_path = Path(it["file"])
        mp4_path = json_path.with_suffix(".mp4")

        if not mp4_path.exists():
            report["missing"].append(
                {
                    "rank": rank,
                    "score": it.get("score"),
                    "json": str(json_path),
                    "mp4": str(mp4_path),
                    "reason": "source_mp4_not_found",
                }
            )
            continue

        if args.flat or (not args.preserve_parent):
            dst = out_dir / mp4_path.name
        else:
            dst = out_dir / mp4_path.parent.name / mp4_path.name

        if dst.exists() and not args.overwrite:
            report["skipped"].append(
                {
                    "rank": rank,
                    "score": it.get("score"),
                    "src": str(mp4_path),
                    "dst": str(dst),
                    "reason": "dst_exists",
                }
            )
            continue

        status = safe_copy(mp4_path, dst, overwrite=args.overwrite)
        report["copied"].append(
            {
                "rank": rank,
                "score": it.get("score"),
                "src": str(mp4_path),
                "dst": str(dst),
                "status": status,
            }
        )

    report["summary"] = {
        "num_planned": len(sliced),
        "num_copied": len(report["copied"]),
        "num_missing": len(report["missing"]),
        "num_skipped": len(report["skipped"]),
    }

    report_path = out_dir / "copy_topn_report.json"
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(
        f"[OK] Copied: {report['summary']['num_copied']}/{report['summary']['num_planned']} -> {out_dir}"
    )
    print(f"[OK] Report: {report_path}")
    if report["missing"]:
        print(f"[WARN] Missing source mp4: {len(report['missing'])}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
