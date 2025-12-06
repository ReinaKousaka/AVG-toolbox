import os
import json
import cv2
from moviepy.editor import VideoFileClip, concatenate_videoclips


def merge_videos_with_frame_label(
    input_dir,
    output_video="merged.mp4",
    output_json="frame_map.json",
    sample_interval=30,
    video_step=1,  # 新增：视频抽帧因子（输出视频只保留每 video_step 帧）
    fps=None,
):
    # 1. 获取视频列表并排序
    video_files = sorted(
        [f for f in os.listdir(input_dir) if f.lower().endswith(".mp4")]
    )
    video_paths = [os.path.join(input_dir, f) for f in video_files]

    if not video_files:
        raise ValueError("No mp4 files found in directory.")

    # 2. 读取所有视频
    clips = []
    for vp in video_paths:
        clip = VideoFileClip(vp)
        clips.append(clip)

    if fps is None:
        fps = clips[0].fps

    # 3. MoviePy 拼接
    merged_clip = concatenate_videoclips(clips, method="compose")

    # 临时文件
    temp_video = "_temp_merged.mp4"
    merged_clip.write_videofile(temp_video, fps=fps, codec="libx264", audio_codec="aac")

    # 4. 第二遍处理：OpenCV 加帧号 + 抽帧 + JSON
    cap = cv2.VideoCapture(temp_video)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Writer 输出较低 FPS（如果按步长抽帧，FPS 应该同步减少）
    output_fps = fps / video_step
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_video, fourcc, output_fps, (w, h))

    frame_map = {}

    # 计算每段视频的帧范围
    segment_ranges = []
    acc = 0
    for clip, name in zip(clips, video_files):
        frames = int(round(clip.duration * fps))
        segment_ranges.append((acc, acc + frames - 1, name))
        acc += frames

    def get_filename_by_frame(f):
        for start, end, name in segment_ranges:
            if start <= f <= end:
                return name
        return None

    global_frame = 0  # 原完整视频的帧计数
    output_frame_count = 0  # 输出视频的帧计数（抽帧后）

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # JSON 抽样记录：按 global_frame 判断
        if global_frame % sample_interval == 0:
            frame_map[str(global_frame)] = get_filename_by_frame(global_frame)

        # 视频抽帧：只写入 global_frame % video_step == 0 的帧
        if global_frame % video_step == 0:
            label = f"{global_frame}"

            write = cv2.resize(frame, (w // 3, h // 3))
            write = cv2.putText(
                write, label, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2
            )
            writer.write(write)
            output_frame_count += 1

        global_frame += 1

    cap.release()
    writer.release()

    # 写 JSON
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(frame_map, f, indent=2, ensure_ascii=False)

    print(f"输出视频: {output_video}")
    print(f"输出帧映射 JSON: {output_json}")
    print(f"原始帧总数: {total_frames}，输出帧数: {output_frame_count}")


# --------------------------
# 示例调用
# --------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_dir", type=str, default="raw_osaka-u_432p", help="输入视频目录"
    )
    parser.add_argument(
        "--output_video",
        type=str,
        default="raw_osaka-u_432p.mp4",
        help="输出合并视频文件名",
    )
    parser.add_argument(
        "--output_json",
        type=str,
        default="raw_osaka-u_432p.json",
        help="输出帧映射 JSON 文件名",
    )
    parser.add_argument(
        "--sample_interval", type=int, default=120, help="JSON 记录的帧间隔"
    )
    parser.add_argument(
        "--video_step", type=int, default=5, help="视频抽帧因子，每 N 帧保留 1 帧"
    )
    args = parser.parse_args()
    merge_videos_with_frame_label(
        input_dir=args.input_dir,
        output_video=args.output_video,
        output_json=args.output_json,
        sample_interval=args.sample_interval,  # JSON 记录间隔
        video_step=args.video_step,  # 抽帧，每 2 帧保留 1 帧
    )
