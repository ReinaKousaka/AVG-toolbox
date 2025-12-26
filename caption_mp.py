import os
import glob
import json
import cv2
import numpy as np
from tqdm import tqdm
from skvideo.io import vwrite
import multiprocessing as mp


def caption_one(prompt_path: str):
    """
    Process one prompt json + its corresponding video.
    Returns (ok: bool, prompt_path: str, msg: str)
    """
    try:
        with open(prompt_path, "r") as f:
            annotations = json.load(f)

        video_path = prompt_path.replace("_prompt_1", "_frustum").replace(
            ".json", "_diff_36x64_nminus40.mp4"
        )
        if not os.path.exists(video_path):
            return False, prompt_path, f"video not found: {video_path}"

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return False, prompt_path, f"failed to open video: {video_path}"

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

        simple_annotation = annotations.get("simple", {})
        text = ""
        font_scale = 0.6
        font_thickness = 1

        captioned_video = []

        # 确保 lines 总是有值（即使当前帧没有新标注）
        lines = []

        for frame_idx in range(total_frames):
            ret, frame = cap.read()
            if not ret:
                break

            # 创建黑色文字区域
            text_area = np.zeros((frame_h, frame_w, 3), dtype=np.uint8)

            # 若该帧有标注，更新文本并重新分行
            key = str(frame_idx)
            if key in simple_annotation:
                read_prompt = "[sep]".join(list(simple_annotation[key].values()))
                if text != read_prompt:
                    text = read_prompt

                # text = text.replace("The video captures ", "")
                # text = text.replace("The video shows ", "")
                if text:
                    text = text[:1].upper() + text[1:]

                # 分行
                max_width = frame_w - 20
                words = text.split()
                new_lines = []
                current_line = ""

                for word in words:
                    test_line = current_line + " " + word if current_line else word
                    (text_width, _), _ = cv2.getTextSize(
                        test_line, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness
                    )
                    if text_width <= max_width:
                        current_line = test_line
                    else:
                        if current_line:
                            new_lines.append(current_line)
                        current_line = word

                if current_line:
                    new_lines.append(current_line)

                lines = new_lines  # 用最新 lines 覆盖（后续帧若无新标注则保持上一段）

            # 绘制文本（如果 lines 为空则什么都不画）
            y_offset = 20
            line_height = int(15 * font_scale / 0.4)

            for line in lines:
                if y_offset + line_height > frame_h:
                    break
                cv2.putText(
                    text_area,
                    line,
                    (10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale,
                    (255, 255, 255),
                    font_thickness,
                    cv2.LINE_AA,
                )
                y_offset += line_height

            # 注意：frame[:, :, ::-1] 是 BGR->RGB（skvideo vwrite 常用 RGB）
            new_frame = np.concatenate((frame[:, :, ::-1], text_area), axis=1)
            captioned_video.append(new_frame)

        cap.release()

        video_basename = os.path.basename(video_path)
        out_path = os.path.join("captioned_videos_mp", video_basename)

        # 写出
        vwrite(out_path, np.asarray(captioned_video))
        return True, prompt_path, out_path

    except Exception as e:
        return False, prompt_path, repr(e)


def main():
    prompts = sorted(glob.glob("raw_2077-12-23_576p_prompt_1/*.json"))
    prompts = list(reversed(prompts))
    os.makedirs("captioned_videos_mp", exist_ok=True)

    # 进程数：默认用 CPU 核心数-1（至少 1）
    nproc = 32
    print(f"Using {nproc} processes for captioning videos.")
    # macOS/Windows 推荐 spawn；Linux 用默认也行
    ctx = mp.get_context("spawn")

    ok_count = 0
    fail_count = 0
    fails = []

    with ctx.Pool(processes=nproc) as pool:
        it = pool.imap_unordered(caption_one, prompts, chunksize=1)
        for ok, prompt_path, msg in tqdm(it, total=len(prompts)):
            if ok:
                ok_count += 1
            else:
                fail_count += 1
                fails.append((prompt_path, msg))

    print(f"\nDone. ok={ok_count}, failed={fail_count}")
    if fails:
        print("\nFailures (up to 20 shown):")
        for p, m in fails[:20]:
            print("-", p, "->", m)


if __name__ == "__main__":
    main()
