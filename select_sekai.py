import csv
import os
import math

# ================== 参数配置 ==================
CSV_PATH = "sekai-game-walking.csv"  # csv 文件路径
JSON_DIR = "sgw-cam"  # json 文件所在文件夹
OUTPUT_PATH = "sekai-good"  # 输出文件
TIME_OF_DAY = "day"  # 只筛选 day
TOP_RATIO = 0.6  # 取前 30%
# =============================================


def get_day_videofiles(csv_path, time_of_day):
    """从 CSV 中筛选 timeOfDay == day 的 videoFile（去掉 .mp4 后缀）"""
    video_names = set()

    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["timeOfDay"].strip().lower() == time_of_day:
                video_file = row["videoFile"].strip()
                base_name = os.path.splitext(video_file)[0]
                video_names.add(base_name)

    return video_names


def collect_json_files(json_dir, valid_video_names):
    """只保留在 videoFiles 中出现的 json，并获取文件大小"""
    json_files_with_size = []

    for name in valid_video_names:
        json_path = os.path.join(json_dir, f"{name}.json")
        if os.path.isfile(json_path):
            size = os.path.getsize(json_path)
            json_files_with_size.append((name, size))

    return json_files_with_size


def select_top_ratio(json_files_with_size, ratio):
    """按文件大小排序并取前 ratio"""
    json_files_with_size.sort(key=lambda x: x[1], reverse=True)
    top_n = math.ceil(len(json_files_with_size) * ratio)
    return json_files_with_size[:top_n]


def save_result(selected_files, output_path):
    """保存文件名到输出文件"""
    with open(output_path, "w", encoding="utf-8") as f:
        for name, size in selected_files:
            f.write(f"{name}\n")


def main():
    # 1. 从 CSV 中获取 day 的 videoFiles
    day_videos = get_day_videofiles(CSV_PATH, TIME_OF_DAY)

    # 2. 收集对应的 json 文件及大小
    json_files = collect_json_files(JSON_DIR, day_videos)

    if not json_files:
        print("没有找到匹配的 JSON 文件")
        return

    # 3. 按大小排序并取前 30%
    top_json_files = select_top_ratio(json_files, TOP_RATIO)

    # 4. 保存结果
    save_result(top_json_files, OUTPUT_PATH)

    print(f"完成：共保存 {len(top_json_files)} 个文件名到 {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
