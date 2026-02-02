#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Convert intrinsics/extrinsics stored in NPZ (key: 'data') into:
- intrinsics: (N,4) -> (N,3,3)
- extrinsics: (N,4,4) c2w -> (N,4,4) w2c
and save as:
  [stem]_intrinsics_da3nested.npy
  [stem]_extrinsics_da3nested.npy
"""

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Tuple, Optional

import numpy as np


def setup_logging(
    log_file: Optional[Path] = None, verbose: bool = False
) -> logging.Logger:
    logger = logging.getLogger("convert_npz_cam")
    logger.setLevel(logging.DEBUG)

    fmt = logging.Formatter(
        "[%(asctime)s][%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )

    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.DEBUG if verbose else logging.INFO)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(str(log_file), encoding="utf-8")
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    return logger


def load_npz_data(npz_path: Path) -> np.ndarray:
    with np.load(str(npz_path)) as z:
        if "data" not in z:
            raise KeyError(f"NPZ missing key 'data': {npz_path}")
        return z["data"]


def intrinsics_4_to_3x3(K4: np.ndarray) -> np.ndarray:
    """
    Expect shape (N,4). Interpret as [fx, fy, cx, cy].
    Return shape (N,3,3).
    """
    if K4.ndim != 2 or K4.shape[1] != 4:
        raise ValueError(f"Intrinsics expected shape (N,4), got {K4.shape}")

    fx = K4[:, 0]
    fy = K4[:, 1]
    cx = K4[:, 2]
    cy = K4[:, 3]

    N = K4.shape[0]
    K = np.zeros((N, 3, 3), dtype=K4.dtype)
    K[:, 0, 0] = fx
    K[:, 1, 1] = fy
    K[:, 0, 2] = cx
    K[:, 1, 2] = cy
    K[:, 2, 2] = 1
    return K


def invert_c2w_to_w2c(T_c2w: np.ndarray) -> np.ndarray:
    """
    Input: (N,4,4) c2w
    Output: (N,4,4) w2c
    Assumes rigid transform.
    """
    if T_c2w.ndim != 3 or T_c2w.shape[1:] != (4, 4):
        raise ValueError(f"Extrinsics expected shape (N,4,4), got {T_c2w.shape}")

    # Rigid inverse: [R t; 0 1]^-1 = [R^T -R^T t; 0 1]
    R = T_c2w[:, :3, :3]
    t = T_c2w[:, :3, 3:4]

    Rt = np.transpose(R, (0, 2, 1))
    t_inv = -(Rt @ t)

    T_w2c = np.zeros_like(T_c2w)
    T_w2c[:, :3, :3] = Rt
    T_w2c[:, :3, 3:4] = t_inv
    T_w2c[:, 3, 3] = 1
    return T_w2c


def find_matching_file(stem: str, ext_dir: Path) -> Optional[Path]:
    """
    Try to find extrinsics/intrinsics NPZ by exact filename match stem + '.npz'.
    If not found, fallback to any single match with same stem (case-sensitive).
    """
    cand = ext_dir / f"{stem}.npz"
    if cand.exists():
        return cand

    # fallback: search for any .npz whose stem matches exactly
    matches = [p for p in ext_dir.glob("*.npz") if p.stem == stem]
    if len(matches) == 1:
        return matches[0]
    return None


def process_pair(
    intri_npz: Path, extri_npz: Path, out_dir: Path, logger: logging.Logger
) -> Tuple[Path, Path]:
    stem = intri_npz.stem

    K4 = load_npz_data(intri_npz)
    T_c2w = load_npz_data(extri_npz)

    K = intrinsics_4_to_3x3(K4)
    T_w2c = invert_c2w_to_w2c(T_c2w)

    out_dir.mkdir(parents=True, exist_ok=True)
    out_ex = out_dir / f"{stem}_extrinsics_da3nested.npy"
    out_in = out_dir / f"{stem}_intrinsics_da3nested.npy"

    np.save(str(out_ex), T_w2c)
    np.save(str(out_in), K)

    logger.debug(f"Saved: {out_ex.name}  shape={T_w2c.shape} dtype={T_w2c.dtype}")
    logger.debug(f"Saved: {out_in.name}  shape={K.shape} dtype={K.dtype}")
    return out_ex, out_in


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--intrinsic_dir",
        "-id",
        type=str,
        required=True,
        help="Directory containing intrinsics .npz",
    )
    p.add_argument(
        "--extrinsic_dir",
        "-ed",
        type=str,
        required=True,
        help="Directory containing extrinsics .npz",
    )
    p.add_argument(
        "--output_dir",
        "-o",
        type=str,
        required=True,
        help="Output directory for .npy files",
    )
    p.add_argument(
        "--strict",
        "-s",
        action="store_true",
        help="If set, missing pairs or shape issues will raise",
    )
    p.add_argument(
        "--verbose", "-v", action="store_true", help="Verbose console logging"
    )
    p.add_argument(
        "--log_dir",
        "-ld",
        type=str,
        default="",
        help="If set, write a log file into this directory",
    )
    return p.parse_args()


def main():
    args = parse_args()
    intri_dir = Path(args.intrinsic_dir)
    extri_dir = Path(args.extrinsic_dir)
    out_dir = Path(args.output_dir)

    log_file = None
    if args.log_dir:
        log_dir = Path(args.log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = log_dir / "convert_npz_cam.log"

    logger = setup_logging(log_file=log_file, verbose=args.verbose)

    if not intri_dir.is_dir():
        raise NotADirectoryError(f"Not a directory: {intri_dir}")
    if not extri_dir.is_dir():
        raise NotADirectoryError(f"Not a directory: {extri_dir}")

    intri_files = sorted(intri_dir.glob("*.npz"))
    if not intri_files:
        logger.warning(f"No .npz found in intrinsic_dir: {intri_dir}")
        return

    total = len(intri_files)
    ok = 0
    skipped = 0
    failed = 0

    logger.info(f"Found {total} intrinsics npz in: {intri_dir}")
    logger.info(f"Extrinsics dir: {extri_dir}")
    logger.info(f"Output dir: {out_dir}")

    for i, intri_npz in enumerate(intri_files, 1):
        stem = intri_npz.stem
        extri_npz = find_matching_file(stem, extri_dir)
        if extri_npz is None:
            msg = f"[{i}/{total}] Missing matching extrinsics for stem='{stem}'"
            if args.strict:
                raise FileNotFoundError(msg)
            logger.warning(msg + " (skip)")
            skipped += 1
            continue

        try:
            logger.info(f"[{i}/{total}] {stem}")
            process_pair(intri_npz, extri_npz, out_dir, logger)
            ok += 1
        except Exception as e:
            if args.strict:
                raise
            logger.exception(f"[{i}/{total}] Failed stem='{stem}': {e}")
            failed += 1

    logger.info(f"Done. ok={ok} skipped={skipped} failed={failed} total={total}")


if __name__ == "__main__":
    main()
