import glob, os
import json, yaml
import numpy as np
import shutil

DA3_DIR = "V_0121_da3"
VIDEO_DIR = "V_0121"
NEW_VIDEO_DIR = "V_0121_left"
os.makedirs(NEW_VIDEO_DIR, exist_ok=True)
if __name__ == "__main__":
    raw_videos = glob.glob(os.path.join(VIDEO_DIR, "*.mp4"))
    # os.makedirs(DA3_DIR, exist_ok=True)
    missing = []
    for raw_video in raw_videos:
        video_basename = os.path.basename(raw_video)[:-4]
        if (
            os.path.exists(
                os.path.join(DA3_DIR, video_basename + "_depth_da3nested.npz")
            )
            and os.path.exists(
                os.path.join(
                    DA3_DIR,
                    video_basename + "_extrinsics_da3nested.npy",
                )
            )
            and os.path.exists(
                os.path.join(
                    DA3_DIR,
                    video_basename + "_intrinsics_da3nested.npy",
                )
            )
        ):
            try:
                depth = np.load(
                    os.path.join(
                        DA3_DIR,
                        video_basename + "_depth_da3nested.npz",
                    )
                )
                extrinsics = np.load(
                    os.path.join(
                        DA3_DIR,
                        video_basename + "_extrinsics_da3nested.npy",
                    )
                )
                intrinsics = np.load(
                    os.path.join(
                        DA3_DIR,
                        video_basename + "_intrinsics_da3nested.npy",
                    )
                )
            except Exception as e:
                print(f"[Error] 视频 {raw_video} 读取失败，原因：{e}，标记为缺失。")
                missing.append(raw_video)
            continue
        else:
            missing.append(raw_video)
            shutil.move(
                raw_video, os.path.join(NEW_VIDEO_DIR, os.path.basename(raw_video))
            )

    with open(os.path.join(VIDEO_DIR, "___missing.yaml"), "w") as f:
        yaml.dump(missing, f)
