import cv2
from tqdm import tqdm
import json
import numpy as np
from skvideo.io import vwrite
import glob, os

if __name__ == "__main__":
    prompts = glob.glob(f"raw_osmo1_576p_prompt/*.json")
    os.makedirs("captioned_videos", exist_ok=True)
    prompts = sorted(prompts)
    prompts = list(reversed(prompts))
    for prompt_path in tqdm(prompts):
        # prompt_path = (
        #     "raw_osmo1_576p_prompt/CAM_20251214164836_0111_D_proc_temp_part_005.json"
        # )
        with open(prompt_path, "r") as f:
            annotations = json.load(f)
        video_path = prompt_path.replace("_prompt", "_frustum").replace(
            ".json", "_diff_36x64_nminus40.mp4"
        )
        video_basename = os.path.basename(video_path)
        if os.path.exists(os.path.join("captioned_videos", video_basename)):
            continue
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_h, frame_w = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(
            cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        )
        simple_annotation = annotations["simple"]
        text = ""
        font_scale = 0.85
        font_thickness = 2
        captioned_video = []
        for frame_idx in range(total_frames):
            ret, frame = cap.read()
            if not ret:
                break

            # 下采样帧
            # frame_resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)

            # 创建黑色文字区域
            text_area = np.zeros((frame_h, frame_w, 3), dtype=np.uint8)

            # 获取当前帧对应的标注
            if f"{frame_idx}" in simple_annotation:
                read_prompt = simple_annotation[f"{frame_idx}"]
                if text != read_prompt:
                    text = read_prompt
                text = text.replace("The video captures ", "")
                text = text.replace("The video shows ", "")
                text = text[:1].upper() + text[1:]  # 首字母大写
                # 将文本分行显示
                max_width = frame_w - 20  # 留出边距
                words = text.split()
                words = words  # 在开头加上帧索引
                lines = []
                current_line = ""

                for word in words:
                    test_line = current_line + " " + word if current_line else word
                    (text_width, text_height), _ = cv2.getTextSize(
                        test_line, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness
                    )

                    if text_width <= max_width:
                        current_line = test_line
                    else:
                        if current_line:
                            lines.append(current_line)
                        current_line = word

                if current_line:
                    lines.append(current_line)

            # 在文字区域绘制文本
            y_offset = 20
            line_height = int(25 * font_scale / 0.4)  # 根据字体缩放调整行高

            for line in lines:
                if y_offset + line_height > frame_h:
                    break  # 超出文字区域

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
            new_frame = np.concatenate((frame[:, :, ::-1], text_area), axis=1)
            captioned_video.append(new_frame)
        video_basename = os.path.basename(video_path)
        vwrite(
            os.path.join("captioned_videos", video_basename),
            np.array(captioned_video),
        )
