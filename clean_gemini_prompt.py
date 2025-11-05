#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
from pathlib import Path
import argparse
from typing import Any, Dict
import os


def parse_json_maybe_double_encoded(text: str) -> Dict[str, Any]:
    """
    尝试把文本解析成 dict：
    - 先 json.loads 一次；
    - 如果得到的是 str，说明是“双重编码”，再 loads 一次；
    - 最终必须是 dict，否则抛错。
    """
    obj = json.loads(text)
    if isinstance(obj, str):
        obj = json.loads(obj)
    if not isinstance(obj, dict):
        raise ValueError("顶层不是 JSON 对象（dict）。")
    return obj


def filter_frames(d: Dict[str, Any], strict: bool = False) -> Dict[str, Any]:
    """
    过滤掉 value 不是 dict 的条目。
    strict=True 时进一步校验：value 必须含有 'first' 和 'remaining' 且为字符串。
    """
    out = {}
    for k, v in d.items():
        if isinstance(v, dict):
            if strict:
                first_ok = isinstance(v.get("first"), str)
                remain_ok = isinstance(v.get("remaining"), str)
                if not (first_ok and remain_ok):
                    continue
            out[k] = v
    return out


def process_file(in_path: Path, out_dir: Path, strict: bool = False) -> None:
    """
    读取单个 JSON 文件，过滤后写入输出目录。
    """
    with open(
        in_path,
        "r",
    ) as f:
        raw_prompt_json = json.load(f)
    for k, v in raw_prompt_json.items():
        if isinstance(v, str):
            print("先用loads")
            try:
                v = json.loads(v)
            except:
                print("试试load")
                try:
                    v = json.load(v)
                except:
                    print(f"最坏情况{k}, {in_path.name}")
                    v = v.split("remaining")
                    v = {
                        "first": v[0][10:].replace('"', "").replace(":", ""),
                        "remaining": v[1][1:].replace('"', "").replace(":", "").replace("}", ""),
                    }
            raw_prompt_json[k] = v
        elif not isinstance(v, dict):
            print(f"{in_path}, 跳过", k)

    # try:
    #     text = in_path.read_text(encoding="utf-8-sig")  # 兼容 BOM
    #     data = parse_json_maybe_double_encoded(text)
    #     filtered = filter_frames(data, strict=strict)
    # except Exception as e:
    #     print(f"[跳过] {in_path.name}: 解析失败 -> {e}")
    #     return
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, in_path.name), "w", encoding="utf-8") as f:
        json.dump(raw_prompt_json, f, indent=4)
    # out_dir.mkdir(parents=True, exist_ok=True)


def main():
    ap = argparse.ArgumentParser(
        description="过滤 JSON 中 value 不是 dict 的 'frame_xxx-frame_xxx' 键，并保存到新目录。"
    )
    ap.add_argument(
        "--input_dir",
        type=str,
        help="输入目录（包含若干 .json 文件）",
        default="gemini_prompt",
    )
    ap.add_argument(
        "--output_dir",
        type=str,
        help="输出目录（保存过滤后的 JSON）",
        default="gemini_prompt_cleaned",
    )
    ap.add_argument(
        "--strict",
        action="store_true",
        help="开启严格校验：value 必须包含字符串类型的 'first' 和 'remaining'。",
    )
    args = ap.parse_args()

    in_dir = Path(args.input_dir)
    out_dir = Path(args.output_dir)

    if not in_dir.exists() or not in_dir.is_dir():
        raise SystemExit(f"输入目录不存在或不是文件夹：{in_dir}")

    files = sorted([p for p in in_dir.glob("*.json") if p.is_file()])
    if not files:
        print(f"输入目录没有 .json 文件：{in_dir}")
        return

    for p in files:
        process_file(p, out_dir, strict=args.strict)


if __name__ == "__main__":
    main()
