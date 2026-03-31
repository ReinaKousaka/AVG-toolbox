import modal
import subprocess
import os
import sys
import glob
import json
import time
from datetime import datetime
import numpy as np

da3_output_volume = modal.Volume.from_name("da3-output", create_if_missing=True)

image = (
    modal.Image.from_registry("python:3.11-slim-bookworm")
    .apt_install("ffmpeg", "libsm6", "libxext6", "git", "build-essential")
    .pip_install(
        "torch",
        "torchvision",
        "numpy",
        "opencv-python",
        "transformers",
        "timm",
        "accelerate",
        "einops",
        "huggingface_hub",
        "xformers",
        "scipy",
        "matplotlib",
        "imageio",
        "trimesh",
        "pygltflib",
        "pillow",
        "tqdm",
        "pyyaml",
        "scikit-learn",
        "scikit-image",
        "plyfile",
        "decord",
        "sk-video",
    )
    .add_local_dir(
        "./depth-anything-3", remote_path="/root/depth-anything-3", copy=True
    )
    .add_local_file(
        "./da3_batched_run_ray.py",
        remote_path="/root/da3_batched_run_ray.py",
        copy=True,
    )
    .add_local_file(
        "./da3nested_lib.py", remote_path="/root/da3nested_lib.py", copy=True
    )
    .add_local_file(
        "./da3nested_lib.py.bak", remote_path="/root/da3nested_lib.py.bak", copy=True
    )
    .run_commands("cd /root/depth-anything-3 && pip install -e .")
)

app = modal.App("da3-batch-processor", image=image)


@app.function(
    volumes={"/videos": modal.Volume.from_name("videos")},
    timeout=60,
)
def list_all_videos():
    videos = []
    for root, dirs, files in os.walk("/videos"):
        for f in files:
            if f.endswith(".mp4"):
                videos.append(os.path.join(root, f))
    return sorted(videos)


def check_output_exists(video_path, output_dir):
    basename = os.path.basename(video_path)
    expected_output = os.path.join(
        output_dir, basename.replace(".mp4", "_depth_da3nested.npz")
    )
    return os.path.exists(expected_output)


@app.function(
    volumes={
        "/videos": modal.Volume.from_name("videos"),
        "/output": da3_output_volume,
    },
    timeout=300,
)
def filter_pending_videos(video_list):
    pending = []
    completed = []
    for v in video_list:
        if check_output_exists(v, "/output"):
            completed.append(os.path.basename(v))
        else:
            pending.append(v)
    return pending, completed


@app.function(
    gpu="B200",
    volumes={
        "/videos": modal.Volume.from_name("videos"),
        "/output": da3_output_volume,
    },
    timeout=14400,
    memory=65536,
)
def process_single_video(video_path: str):
    import sys

    sys.path.insert(0, "/root")
    sys.path.insert(0, "/root/depth-anything-3")

    os.chdir("/root")
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    basename = os.path.basename(video_path)
    video_dir = os.path.dirname(video_path)

    print(f"\n{'='*60}")
    print(f"[START] {basename}")
    print(f"{'='*60}")

    if check_output_exists(video_path, "/output"):
        print(f"[SKIP] Already done: {video_path}")
        return {"status": "skipped", "video": basename}

    cmd = [
        "python",
        "da3_batched_run_ray.py",
        "--input_dirs",
        video_dir,
        "--output_dir",
        "/output",
        "--chunk_size",
        "500",
        "--pose_overlap",
        "1",
        "--process_res",
        "700",
        "--model_name",
        "depth-anything/DA3NESTED-GIANT-LARGE-1.1",
    ]

    print(f"[CMD] {' '.join(cmd)}")
    start_time = time.time()

    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
    )

    while True:
        line = process.stdout.readline()
        if line:
            print(f"[OUT] {line.rstrip()}")
            sys.stdout.flush()
        elif process.poll() is not None:
            break

    for line in process.stderr:
        print(f"[ERR] {line.rstrip()}", file=sys.stderr)
        sys.stderr.flush()

    returncode = process.wait()
    elapsed = time.time() - start_time

    # commit changes to the volume
    da3_output_volume.commit()

    exists = check_output_exists(video_path, "/output")
    status = "success" if (returncode == 0 and exists) else "failed"

    print(f"[{status.upper()}] {basename} in {elapsed:.1f}s")

    return {
        "status": status,
        "video": basename,
        "time": elapsed,
        "returncode": returncode,
    }


@app.local_entrypoint()
def main():
    print("=" * 60)
    print("DA3 Batch Processor - Starting")
    print("=" * 60)

    print("\n[1/4] Scanning videos...")
    all_videos = list_all_videos.remote()
    print(f"Found {len(all_videos)} videos")

    if not all_videos:
        print("No videos found!")
        return

    print("\n[2/4] Checking existing outputs...")
    pending_videos, completed = filter_pending_videos.remote(all_videos)

    print(f"Completed: {len(completed)}")
    for name in completed[:5]:
        print(f"  ✓ {name}")
    if len(completed) > 5:
        print(f"  ... and {len(completed)-5} more")

    print(f"\nPending: {len(pending_videos)}")
    for v in pending_videos[:5]:
        print(f"  • {os.path.basename(v)}")
    if len(pending_videos) > 5:
        print(f"  ... and {len(pending_videos)-5} more")

    if not pending_videos:
        print("\n✓ All videos already processed!")
        return

    print(f"\n[3/4] Processing {len(pending_videos)} videos...")

    batch_size = 10
    total = len(pending_videos)
    all_results = []

    for i in range(0, total, batch_size):
        batch = pending_videos[i : i + batch_size]
        batch_num = i // batch_size + 1
        total_batches = (total + batch_size - 1) // batch_size

        print(f"\n{'='*60}")
        print(f"Batch {batch_num}/{total_batches} ({len(batch)} videos)")
        print(f"{'='*60}")

        # parallely
        results = list(process_single_video.map(batch))
        all_results.extend(results)

        success = sum(1 for r in results if r["status"] == "success")
        skipped = sum(1 for r in results if r["status"] == "skipped")
        failed = sum(1 for r in results if r["status"] == "failed")

        print(f"\nBatch {batch_num} done: {success}✓ {skipped}⊘ {failed}✗")

    print(f"\n[4/4] Final Summary")
    print(f"{'='*60}")
    total_success = sum(1 for r in all_results if r["status"] == "success")
    total_skipped = sum(1 for r in all_results if r["status"] == "skipped")
    total_failed = sum(1 for r in all_results if r["status"] == "failed")

    print(f"Success:   {total_success}")
    print(f"Skipped:   {total_skipped}")
    print(f"Failed:    {total_failed}")
    print(f"Total:     {len(all_results)}")
    print(f"{'='*60}")

    if total_failed > 0:
        print("\nFailed videos:")
        for r in all_results:
            if r["status"] == "failed":
                print(f"  ✗ {r['video']}")
