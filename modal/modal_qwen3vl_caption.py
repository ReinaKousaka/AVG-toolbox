import modal
import os
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import tempfile
import json
import math
from datetime import datetime

# modal image
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg", "libgl1-mesa-glx", "libglib2.0-0", "libsm6", "libxext6")
    .pip_install(
        "hf-transfer",
        "torch>=2.0.0",
        "torchvision",
        "transformers>=4.57.0",
        "qwen-vl-utils",
        "accelerate>=0.26.0",
        "opencv-python-headless",
        "numpy",
        "tqdm",
        "scikit-video",
        "pillow",
        "av",
        "decord",
    )
)

videos_volume = modal.Volume.from_name("youtube-videos", create_if_missing=True)
output_volume = modal.Volume.from_name("captions-output", create_if_missing=True)
model_cache_volume = modal.Volume.from_name("model-cache", create_if_missing=True)

hf_secret = modal.Secret.from_name("huggingface-secret")

app = modal.App("qwen3vl-video-caption", image=image)


GPU_MAP = {
    "a100": "A100-80GB",
    "a100-40gb": "A100-40GB",
    "h100": "H100",
    "a10g": "A10G",
    "t4": "T4",
}


# def get_gpu(gpu_name: str = "a100"):
#     return GPU_MAP.get(gpu_name.lower(), "A100-80GB")
def get_gpu(gpu_name: str = "h100"):
    return GPU_MAP.get(gpu_name.lower(), "H100")


def get_video_info(video_path: str) -> Dict:
    """
    获取视频信息，使用多种方法尝试
    """
    import cv2

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        return {"error": f"Cannot open video: {video_path}"}

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    cap.release()

    # 检查是否获取到有效信息
    if total_frames == 0:
        # 尝试用 ffprobe 获取信息
        try:
            import subprocess

            result = subprocess.run(
                [
                    "ffprobe",
                    "-v",
                    "error",
                    "-select_streams",
                    "v:0",
                    "-count_frames",
                    "-show_entries",
                    "stream=nb_read_frames,r_frame_rate,width,height",
                    "-of",
                    "json",
                    str(video_path),
                ],
                capture_output=True,
                text=True,
                timeout=30,
            )
            if result.returncode == 0:
                info = json.loads(result.stdout)
                stream = info.get("streams", [{}])[0]
                total_frames = int(stream.get("nb_read_frames", 0))
                width = int(stream.get("width", 0))
                height = int(stream.get("height", 0))
                # 解析帧率分数
                fps_str = stream.get("r_frame_rate", "30/1")
                if "/" in fps_str:
                    num, den = fps_str.split("/")
                    fps = float(num) / float(den) if float(den) != 0 else 30.0
                else:
                    fps = float(fps_str)
        except Exception as e:
            print(f"ffprobe failed: {e}")

    return {
        "total_frames": total_frames,
        "fps": fps if fps and fps > 0 else 30.0,
        "width": width,
        "height": height,
        "valid": total_frames > 0,
    }


def extract_frames_robust(
    video_path: str, frame_indices: List[int], downscale_ratio: float = 0.5
) -> Dict[int, any]:
    """
    鲁棒的帧提取，支持多种方法
    """
    import cv2
    import numpy as np

    # 首先尝试 OpenCV
    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        print(
            f"Warning: OpenCV cannot open {video_path}, trying alternative methods..."
        )
        return {}

    extracted = {}
    failed_indices = []

    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()

        if ok and frame is not None and frame.size > 0:
            h, w = frame.shape[:2]
            new_w = max(1, int(w * downscale_ratio) // 2 * 2)
            new_h = max(1, int(h * downscale_ratio) // 2 * 2)
            extracted[idx] = cv2.resize(
                frame, (new_w, new_h), interpolation=cv2.INTER_AREA
            )
        else:
            failed_indices.append(idx)

    cap.release()

    # 如果有失败的帧，尝试用 PyAV
    if failed_indices:
        try:
            import av

            container = av.open(str(video_path))
            video_stream = container.streams.video[0]

            # 计算时间戳
            fps = (
                float(video_stream.average_rate) if video_stream.average_rate else 30.0
            )

            for idx in failed_indices:
                if idx in extracted:
                    continue

                timestamp = int(idx / fps * av.time_base)
                container.seek(timestamp, stream=video_stream)

                for frame in container.decode(video_stream):
                    img = frame.to_ndarray(format="bgr24")
                    h, w = img.shape[:2]
                    new_w = max(1, int(w * downscale_ratio) // 2 * 2)
                    new_h = max(1, int(h * downscale_ratio) // 2 * 2)
                    extracted[idx] = cv2.resize(
                        img, (new_w, new_h), interpolation=cv2.INTER_AREA
                    )
                    break  # 只取一帧

            container.close()
        except Exception as e:
            print(f"PyAV fallback failed: {e}")

    return extracted


# ============ 进度检查功能 ============
@app.function(
    volumes={
        "/mnt/youtube-videos": videos_volume,
        "/mnt/captions-output": output_volume,
    },
)
def check_progress(
    input_dir: str = "/mnt/youtube-videos",
    output_dir: str = "/mnt/captions-output",
    pattern: str = "*.mp4",
    show_details: bool = False,
    max_details: int = 20,
) -> Dict:
    """检查处理进度"""
    input_path = Path(input_dir)
    output_path = Path(output_dir)

    video_files = sorted(input_path.glob(pattern))
    total = len(video_files)

    if total == 0:
        return {"error": f"No videos found in {input_dir}"}

    completed = []
    pending = []
    errors = []

    for video_file in video_files:
        output_json = output_path / f"{video_file.stem}.json"

        if output_json.exists():
            try:
                with open(output_json, "r") as f:
                    data = json.load(f)

                blocks = data.get("blocks", [])
                detailed = data.get("detailed", {})

                if blocks and detailed:
                    completed.append(
                        {
                            "video": video_file.name,
                            "blocks": len(blocks),
                            "frames": len(detailed),
                            "modified": datetime.fromtimestamp(
                                output_json.stat().st_mtime
                            ).strftime("%m-%d %H:%M"),
                        }
                    )
                else:
                    errors.append({"video": video_file.name, "issue": "Empty output"})
            except Exception as e:
                errors.append({"video": video_file.name, "error": str(e)})
        else:
            pending.append(video_file.name)

    print(f"\n{'='*70}")
    print(f"📊 处理进度报告")
    print(f"{'='*70}")
    print(f"总视频数:     {total:4d}")
    print(f"✅ 已完成:     {len(completed):4d} ({round(len(completed)/total*100, 1)}%)")
    print(f"⏳ 待处理:     {len(pending):4d}")
    print(f"❌ 问题:       {len(errors):4d}")
    print(f"{'='*70}")

    if completed and show_details:
        print(f"\n📁 已完成 (前{min(len(completed), max_details)}个):")
        for i, c in enumerate(completed[:max_details], 1):
            print(f"  {i}. {c['video']:<30} {c['blocks']:>3} blocks")

    if pending and show_details:
        print(f"\n📝 待处理 (前{min(len(pending), max_details)}个):")
        for i, p in enumerate(pending[:max_details], 1):
            print(f"  {i}. {p}")

    return {
        "total": total,
        "completed": len(completed),
        "pending": len(pending),
        "progress": round(len(completed) / total * 100, 1),
    }


# pre-download models
@app.function(
    gpu=get_gpu("h100"),
    timeout=3600,
    volumes={"/root/.cache/huggingface": model_cache_volume},
    secrets=[hf_secret],
)
def download_model(model_id: str = "Qwen/Qwen3-VL-30B-A3B-Instruct"):
    """预下载模型"""
    import os

    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

    from transformers import (
        AutoProcessor,
        Qwen3VLMoeForConditionalGeneration,
        Qwen3VLForConditionalGeneration,
    )

    print(f"Downloading {model_id}...")

    processor = AutoProcessor.from_pretrained(
        model_id, cache_dir="/root/.cache/huggingface", trust_remote_code=True
    )

    if "A3B" in model_id:
        model = Qwen3VLMoeForConditionalGeneration.from_pretrained(
            model_id,
            cache_dir="/root/.cache/huggingface",
            torch_dtype="auto",
            device_map="auto",
            trust_remote_code=True,
        )
    else:
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_id,
            cache_dir="/root/.cache/huggingface",
            torch_dtype="auto",
            device_map="auto",
            trust_remote_code=True,
        )

    model_cache_volume.commit()
    print("✓ Download complete")
    return {"status": "success"}


# handles block
@app.cls(
    gpu=get_gpu("h100"),
    memory=65536,
    timeout=1800,
    scaledown_window=300,
    volumes={
        "/mnt/captions-output": output_volume,
        "/root/.cache/huggingface": model_cache_volume,
    },
    secrets=[hf_secret],
)
class BlockProcessor:
    """并行处理单个block"""

    model_id: str = modal.parameter(default="Qwen/Qwen3-VL-30B-A3B-Instruct")
    max_tokens: int = modal.parameter(default=150)

    @modal.enter()
    def load_model(self):
        import torch
        from transformers import (
            Qwen3VLForConditionalGeneration,
            AutoProcessor,
            Qwen3VLMoeForConditionalGeneration,
        )

        if "A3B" in self.model_id:
            self.model = Qwen3VLMoeForConditionalGeneration.from_pretrained(
                self.model_id,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                cache_dir="/root/.cache/huggingface",
                trust_remote_code=True,
            )
        else:
            self.model = Qwen3VLForConditionalGeneration.from_pretrained(
                self.model_id,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                cache_dir="/root/.cache/huggingface",
                trust_remote_code=True,
            )
        self.processor = AutoProcessor.from_pretrained(
            self.model_id, cache_dir="/root/.cache/huggingface", trust_remote_code=True
        )

    def _summarize(self, frames_bgr, prompt, num_images=5):
        import torch
        import cv2
        from PIL import Image

        if not frames_bgr:
            return ""

        with tempfile.TemporaryDirectory() as tmpdir:
            contents = []
            for i, frame in enumerate(frames_bgr[:num_images]):
                img_path = Path(tmpdir) / f"f_{i}.jpg"
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                Image.fromarray(frame_rgb).save(str(img_path))
                contents.append({"type": "image", "image": str(img_path)})

            contents.append({"type": "text", "text": prompt})
            messages = [{"role": "user", "content": contents}]

            inputs = self.processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
            )
            inputs.pop("token_type_ids", None)
            inputs = inputs.to(self.model.device)

            with torch.no_grad():
                generated_ids = self.model.generate(
                    **inputs, max_new_tokens=self.max_tokens, do_sample=False
                )

            trimmed = [
                out[len(in_) :] for in_, out in zip(inputs["input_ids"], generated_ids)
            ]
            result = self.processor.batch_decode(
                trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
            return str(result[0]).replace("\n", " ").replace("\r", " ")

    @modal.method()
    def process_block(self, block_data: Dict) -> Dict:
        import base64
        import cv2
        import numpy as np
        from PIL import Image
        import io

        block_idx = block_data["block_idx"]
        block_start = block_data["block_start"]
        total_frames = block_data["total_frames"]

        frames = []
        for b64 in block_data["frames_base64"]:
            img_bytes = base64.b64decode(b64)
            img = Image.open(io.BytesIO(img_bytes))
            frames.append(cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR))

        if len(frames) < 2:
            return {"block_idx": block_idx, "error": "insufficient frames"}

        static_prompt = """Please summarize the content of the image. Provide a concise within 130 English words. 
Most words should describe the objects in the video in detail, including their shape, color, 
texture & material and words if visible, as well as their approximate location in the frame. 
Then use only very few words to briefly describe the atmosphere, lighting, and other overall 
information of the image. Avoid starting with lengthy phrases such as [in the picture] or 
[this is a picture] Get straight to describing the objects."""

        dynamic_template = """Referring to the description of the first frame description: 
[start first frame description]{static}[end first frame description], 
describe the appearance and movement of objects in the video within 130 English words. 
DO NOT include descriptions of the overall video information such as moods, lights and atmosphere. 
DO NOT start with lengthy phrases such as [in the subsequent frame] or [in the video], 
get straight to describing the objects and movements. Also, when no moving object is observed 
in the scene, DO NOT write [no movement is observed] or [the scene is static] or 
[xxx remain static / still] or any similar sentences, instead, describe the scene in more 
detailed appearances. Assume common world knowledge: Buildings, roads, and large structures 
of the scene are static by default. Do not explicitly state their lack of motion, describe 
MORE about their appearances. DO NOT include any words related to camera."""

        static_caption = self._summarize([frames[0]], static_prompt, num_images=1)
        dynamic_caption = self._summarize(
            frames,
            dynamic_template.format(static=static_caption),
            num_images=len(frames),
        )

        return {
            "block_idx": block_idx,
            "block_start": block_start,
            "block_end": min(block_start + 80, total_frames - 1),
            "frame_indices": block_data["frame_indices"],
            "start_caption": static_caption,
            "dynamic_caption": dynamic_caption,
        }


# video processor
@app.cls(
    gpu=get_gpu("h100"),
    memory=65536,
    timeout=7200,
    scaledown_window=600,
    volumes={
        "/mnt/youtube-videos": videos_volume,
        "/mnt/captions-output": output_volume,
        "/root/.cache/huggingface": model_cache_volume,
    },
    secrets=[hf_secret],
)
class Qwen3VLProcessor:
    model_id: str = modal.parameter(default="Qwen/Qwen3-VL-30B-A3B-Instruct")
    block_size: int = modal.parameter(default=80)
    frames_per_block: int = modal.parameter(default=5)
    downscale_ratio_str: str = modal.parameter(default="0.5")
    max_tokens: int = modal.parameter(default=150)

    @modal.enter()
    def setup(self):
        self.downscale_ratio = float(self.downscale_ratio_str)
        self.frame_offsets = [
            int(i * (self.block_size / (self.frames_per_block - 1)))
            for i in range(self.frames_per_block)
        ]

    @modal.enter()
    def load_model(self):
        import torch
        from transformers import (
            Qwen3VLForConditionalGeneration,
            AutoProcessor,
            Qwen3VLMoeForConditionalGeneration,
        )

        if "A3B" in self.model_id:
            self.model = Qwen3VLMoeForConditionalGeneration.from_pretrained(
                self.model_id,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                cache_dir="/root/.cache/huggingface",
                trust_remote_code=True,
            )
        else:
            self.model = Qwen3VLForConditionalGeneration.from_pretrained(
                self.model_id,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                cache_dir="/root/.cache/huggingface",
                trust_remote_code=True,
            )
        self.processor = AutoProcessor.from_pretrained(
            self.model_id, cache_dir="/root/.cache/huggingface", trust_remote_code=True
        )

    def _get_blocks(self, total_frames):
        blocks = []
        num_blocks = math.ceil(total_frames / self.block_size)
        for block_idx in range(num_blocks):
            block_start = block_idx * self.block_size
            block_end = min(block_start + self.block_size, total_frames - 1)
            indices = []
            for offset in self.frame_offsets:
                idx = block_start + offset
                if idx <= block_end:
                    indices.append(idx)
                elif indices:
                    indices.append(block_end)
            seen = set()
            unique = [
                x
                for x in indices
                if not (x in seen or seen.add(x)) and x < total_frames
            ]
            if unique:
                blocks.append((block_idx, block_start, unique))
        return blocks

    def _summarize(self, frames, prompt, num_images=5):
        import torch
        import cv2
        from PIL import Image

        if not frames:
            return ""

        with tempfile.TemporaryDirectory() as tmpdir:
            contents = []
            for i, frame in enumerate(frames[:num_images]):
                img_path = Path(tmpdir) / f"f_{i}.jpg"
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                Image.fromarray(frame_rgb).save(str(img_path))
                contents.append({"type": "image", "image": str(img_path)})

            contents.append({"type": "text", "text": prompt})
            messages = [{"role": "user", "content": contents}]

            inputs = self.processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
            )
            inputs.pop("token_type_ids", None)
            inputs = inputs.to(self.model.device)

            with torch.no_grad():
                generated_ids = self.model.generate(
                    **inputs, max_new_tokens=self.max_tokens, do_sample=False
                )

            trimmed = [
                out[len(in_) :] for in_, out in zip(inputs["input_ids"], generated_ids)
            ]
            result = self.processor.batch_decode(
                trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
            return str(result[0]).replace("\n", " ").replace("\r", " ")

    @modal.method()
    def process_video(
        self, video_path: str, output_dir: str = "/mnt/captions-output"
    ) -> Dict:
        import cv2
        import numpy as np
        from tqdm import tqdm

        video_file = Path(video_path)
        output_json = Path(output_dir) / f"{video_file.stem}.json"

        # duplicate check
        if output_json.exists():
            print(f"⚠️  Already exists: {output_json}, skipping")
            return {"status": "skipped", "output": str(output_json)}

        if not video_file.exists():
            print(f"❌ Video file not found: {video_path}")
            return {"status": "error", "error": "File not found"}

        print(f"\n{'='*70}")
        print(f"📹 Processing: {video_file.name}")
        print(f"{'='*70}")

        info = get_video_info(str(video_file))

        if "error" in info:
            print(f"❌ Failed to get video info: {info['error']}")
            return {"status": "error", "error": info["error"]}

        if not info["valid"]:
            print(f"❌ Invalid video: 0 frames detected")
            print(f"   Trying alternative methods...")
            # 尝试用 ffprobe 直接获取
            try:
                import subprocess

                result = subprocess.run(
                    [
                        "ffprobe",
                        "-v",
                        "quiet",
                        "-print_format",
                        "json",
                        "-show_format",
                        "-show_streams",
                        str(video_file),
                    ],
                    capture_output=True,
                    text=True,
                )
                print(f"   ffprobe output: {result.stdout[:500]}")
            except Exception as e:
                print(f"   ffprobe failed: {e}")

            return {"status": "error", "error": "Could not read video frames"}

        total_frames = info["total_frames"]
        fps = info["fps"]
        width = info["width"]
        height = info["height"]

        print(f"✓ Video info: {total_frames} frames, {width}x{height}, {fps:.2f} fps")

        # 计算 blocks
        blocks = self._get_blocks(total_frames)
        print(f"✓ Will process {len(blocks)} blocks")

        if len(blocks) == 0:
            return {"status": "error", "error": "No blocks to process"}

        # 收集所有需要的帧索引
        all_indices = set()
        for _, _, indices in blocks:
            all_indices.update(indices)
        all_indices = sorted(list(all_indices))

        print(f"✓ Need to extract {len(all_indices)} unique frames")

        # 提取帧（使用修复版函数）
        print(f"\n⏳ Extracting frames...")
        extracted = extract_frames_robust(
            str(video_file), all_indices, self.downscale_ratio
        )

        print(f"✓ Successfully extracted {len(extracted)}/{len(all_indices)} frames")

        if len(extracted) == 0:
            return {"status": "error", "error": "Could not extract any frames"}

        # 处理 blocks
        print(f"\n⏳ Captioning {len(blocks)} blocks...")
        annotations = {"detailed": {}, "blocks": []}

        for block_idx, block_start, frame_indices in tqdm(blocks, desc="Captioning"):
            # 获取这个 block 的帧
            frames = []
            for idx in frame_indices:
                if idx in extracted:
                    frames.append(extracted[idx])

            if len(frames) < 2:
                print(
                    f"⚠️  Block {block_idx}: insufficient frames ({len(frames)}), skipping"
                )
                continue

            # 生成 captions
            static = self._summarize(
                [frames[0]],
                "Describe objects in detail within 130 words. Avoid phrases like 'in the picture'.",
                1,
            )
            dynamic = self._summarize(
                frames,
                f"Referring to: {static}. Describe movement within 130 words. No camera mentions.",
                len(frames),
            )

            annotations["blocks"].append(
                {
                    "block_idx": block_idx,
                    "block_start_frame": block_start,
                    "block_end_frame": min(
                        block_start + self.block_size, total_frames - 1
                    ),
                    "frame_indices": frame_indices,
                    "start_caption": static,
                    "dynamic_caption": dynamic,
                }
            )
            print(f"Video: {video_file} | Block {block_idx}")
            print(f"Frames: {frame_indices}")
            print(f"Static Caption:\n{static}")
            print(f"Dynamic Caption:\n{dynamic}")
            print("-" * 60)

            stride = 20
            end = min(block_start + self.block_size, total_frames - 1)

            for fidx in range(block_start, end + 1, stride):
                annotations["detailed"][str(fidx)] = {
                    "start": static,
                    "dynamic": dynamic,
                }

            # save results
            if block_idx % 10 == 0:
                Path(output_dir).mkdir(parents=True, exist_ok=True)
                with open(output_json, "w", encoding="utf-8") as f:
                    json.dump(annotations, f, ensure_ascii=False, indent=2)

                output_volume.commit()

        print(f"\n{'='*70}")
        print(f"✅ Completed: {video_file.name}")
        print(f"   Blocks: {len(annotations['blocks'])}/{len(blocks)}")
        print(f"   Output: {output_json}")
        print(f"{'='*70}")

        return {
            "status": "success",
            "video": str(video_file),
            "blocks": len(annotations["blocks"]),
            "total_blocks": len(blocks),
        }


# ============ 并行处理 ============


@app.function(
    gpu=get_gpu("h100"),
    memory=65536,
    timeout=3600,
    volumes={
        "/mnt/youtube-videos": videos_volume,
        "/mnt/captions-output": output_volume,
    },
)
def process_video_parallel(
    video_path: str,
    output_dir: str = "/mnt/captions-output",
    model_id: str = "Qwen/Qwen3-VL-30B-A3B-Instruct",
    block_size: int = 80,
    frames_per_block: int = 5,
    downscale_ratio: float = 0.5,
    max_tokens: int = 150,
    max_parallel: int = 8,
) -> Dict:
    import cv2
    import numpy as np
    from tqdm import tqdm
    import base64
    from PIL import Image
    import io

    video_file = Path(video_path)
    output_json = Path(output_dir) / f"{video_file.stem}.json"

    if output_json.exists():
        return {"status": "skipped", "output": str(output_json)}

    # 获取视频信息
    info = get_video_info(str(video_file))

    if "error" in info or not info["valid"]:
        return {"status": "error", "error": info.get("error", "Invalid video")}

    total_frames = info["total_frames"]
    fps = info["fps"]

    print(f"\nVideo: {video_file.name}")
    print(f"  {total_frames} frames, {fps:.2f} fps")

    # 计算 blocks
    frame_offsets = [
        int(i * (block_size / (frames_per_block - 1))) for i in range(frames_per_block)
    ]
    blocks_meta = []

    for block_idx in range(math.ceil(total_frames / block_size)):
        block_start = block_idx * block_size
        block_end = min(block_start + block_size, total_frames - 1)
        indices = []
        for offset in frame_offsets:
            idx = block_start + offset
            if idx <= block_end:
                indices.append(idx)
            elif indices:
                indices.append(block_end)
        seen = set()
        unique = [
            x for x in indices if not (x in seen or seen.add(x)) and x < total_frames
        ]
        if unique:
            blocks_meta.append((block_idx, block_start, unique))

    print(f"  {len(blocks_meta)} blocks")

    # 提取帧
    all_indices = set()
    for _, _, indices in blocks_meta:
        all_indices.update(indices)
    all_indices = sorted(list(all_indices))

    print(f"Extracting {len(all_indices)} frames...")
    extracted_frames = extract_frames_robust(
        str(video_file), all_indices, downscale_ratio
    )

    print(f"  Extracted {len(extracted_frames)} frames")

    if len(extracted_frames) == 0:
        return {"status": "error", "error": "Could not extract frames"}

    # 编码为 base64
    def encode_frame(frame):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame_rgb)
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG", quality=85)
        return base64.b64encode(buffer.getvalue()).decode("utf-8")

    block_data_list = []
    for block_idx, block_start, indices in blocks_meta:
        frames = [extracted_frames[i] for i in indices if i in extracted_frames]
        if len(frames) >= 2:
            block_data_list.append(
                {
                    "block_idx": block_idx,
                    "block_start": block_start,
                    "frame_indices": indices,
                    "frames_base64": [encode_frame(f) for f in frames],
                    "total_frames": total_frames,
                }
            )

    print(
        f"Processing {len(block_data_list)} blocks with max_parallel={max_parallel}..."
    )

    processor = BlockProcessor(model_id=model_id, max_tokens=max_tokens)

    all_results = []
    for i in range(0, len(block_data_list), max_parallel):
        batch = block_data_list[i : i + max_parallel]
        print(f"  Batch {i//max_parallel + 1}: {len(batch)} blocks")

        batch_results = list(processor.process_block.map(batch, order_outputs=False))
        all_results.extend(batch_results)
        print(f"    ✓ {len([r for r in batch_results if 'error' not in r])} succeeded")

    # 合并结果
    annotations = {"detailed": {}, "blocks": []}

    stride = 20
    for result in sorted(all_results, key=lambda x: x.get("block_idx", 0)):
        if "error" in result:
            continue

        annotations["blocks"].append(result)

        block_start = result["block_start"]
        end = min(block_start + block_size, total_frames - 1)

        for fidx in range(block_start, end + 1, stride):
            annotations["detailed"][str(fidx)] = {
                "start": result["start_caption"],
                "dynamic": result["dynamic_caption"],
            }

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(annotations, f, ensure_ascii=False, indent=2)
    output_volume.commit()

    return {
        "status": "success",
        "video": str(video_file),
        "total_blocks": len(blocks_meta),
        "processed_blocks": len(annotations["blocks"]),
    }


@app.function(volumes={"/mnt/youtube-videos": videos_volume})
def list_videos_cloud(input_dir: str, pattern: str) -> List[str]:
    import os
    from pathlib import Path

    p = Path(input_dir)
    return [str(f) for f in sorted(p.glob(pattern)) if f.is_file()]


@app.local_entrypoint()
def main(
    mode: str = "check",
    input_dir: str = "/mnt/youtube-videos",
    output_dir: str = "/mnt/captions-output",
    video: Optional[str] = None,
    model_id: str = "Qwen/Qwen3-VL-30B-A3B-Instruct",
    pattern: str = "*.mp4",
    max_videos: Optional[int] = None,
    gpu: str = "a100",
    block_size: int = 80,
    frames_per_block: int = 5,
    downscale_ratio: str = "0.5",
    max_tokens: int = 150,
    max_parallel: int = 8,
    show_details: bool = False,
    max_details: int = 20,
    block: int = 0,
):
    if mode == "check":
        result = check_progress.remote(
            input_dir=input_dir,
            output_dir=output_dir,
            pattern=pattern,
            show_details=show_details,
            max_details=max_details,
        )
        return result
    elif mode == "download":
        print(f"Downloading model {model_id}...")
        result = download_model.remote(model_id)
        print(f"Result: {result}")
        return result
    elif mode == "process":
        if video:
            print(f"Processing single video: {video} with GPU={gpu}")
            processor = Qwen3VLProcessor(
                model_id=model_id,
                block_size=block_size,
                frames_per_block=frames_per_block,
                downscale_ratio_str=downscale_ratio,
                max_tokens=max_tokens,
            )
            result = processor.process_video.remote(video, output_dir)
            print(f"\nResult: {result.get('status')}")
            return result
        else:
            print(f"Fetching video list from cloud Volume: {input_dir}")
            video_files = list_videos_cloud.remote(input_dir, pattern)
            if max_videos:
                video_files = video_files[:max_videos]
            if not video_files:
                print("No videos found")
                return
            print(f"Batch processing {len(video_files)} videos with GPU={gpu}")

            processor = Qwen3VLProcessor(
                model_id=model_id,
                block_size=block_size,
                frames_per_block=frames_per_block,
                downscale_ratio_str=downscale_ratio,
                max_tokens=max_tokens,
            )

            results = []
            for i, vf in enumerate(video_files, 1):
                vf_path = Path(vf)
                print(f"\n[{i}/{len(video_files)}] {vf_path.name}")
                try:
                    r = processor.process_video.remote(str(vf), output_dir)
                    results.append(r)
                    print(f"  → {r.get('status')}")
                except Exception as e:
                    print(f"  ✗ Error: {e}")
                    results.append(
                        {"status": "error", "video": str(vf), "error": str(e)}
                    )

            success = [r for r in results if r.get("status") == "success"]
            print(f"\nDone: {len(success)}/{len(results)} success")
            return results

    elif mode == "parallel":
        if not video:
            print("Error: --video required for parallel mode")
            return

        print(f"Parallel processing: {video}")
        result = process_video_parallel.remote(
            video_path=video,
            output_dir=output_dir,
            model_id=model_id,
            block_size=block_size,
            frames_per_block=frames_per_block,
            downscale_ratio=float(downscale_ratio),
            max_tokens=max_tokens,
            max_parallel=max_parallel,
        )

        print(f"\nResult: {result.get('status')}")
        return result

    else:
        print(f"Unknown mode: {mode}")
