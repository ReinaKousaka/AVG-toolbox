"""
Author: Songheng Yin, songheng.yin@columbia.edu
Use OpenAI API, to generate captions given video input.
Note that Google sets API rates: https://ai.google.dev/gemini-api/docs/rate-limits
The script uses backoff to handle it

Setup:
pip install -U -q "google-genai"
pip install backoff

Example Usage:
time python run_gemini_api_for_caption --snippet_length 128 --frame_gap 16 --video_dir some_dir --output_dir result_dir
"""

import argparse
import base64
import os
from pathlib import Path
from tqdm import tqdm
import logging
import backoff
import cv2
import av  # for .hevc

from google import genai
from google.genai import types
# from google.api_core.exceptions import GoogleAPIError


# FILL THIS
api_key = "YOUR_GEMINI_API_KEY_HERE"

parser = argparse.ArgumentParser()
parser.add_argument("--snippet_length", type=int, default=96, help="snippet length of subvideo to caption")
parser.add_argument("--frame_gap", type=int, default=16, help="frame gap to feed")
parser.add_argument("--video_dir", type=str, default="", help="input path of video")
parser.add_argument("--output_dir", type=str, default="caption-gemini-out", help="output path")

args = parser.parse_args()
SNIPPET_LENGTH = args.snippet_length
FRMAE_GAP = args.frame_gap
INPUT_DIR = args.video_dir
OUTPUT_DIR = args.output_dir
assert SNIPPET_LENGTH % FRMAE_GAP == 0



logging.basicConfig(
    level=logging.INFO,  # mute Google's DEBUG level messages
    format='[%(asctime)s] - %(levelname)s: %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

# CONFIDENTIAL KEY! Better to use env variable by:
# export GEMINI_API_KEY="XYZ"
# client = genai.Client(api_key=api_key)


def _format(number: int) -> str:
    return f"frame_{str(number).zfill(10)}"


def backoff_logger(details):
    """logger for backoff"""
    tries = details['tries']
    if tries < 5:
        logger.info(f'Retry {tries} times')
    else:
        logger.warning(f'Retry {tries} times')

# wrapper to retry on failures
@backoff.on_exception(
    backoff.constant,
    Exception,
    interval=60*20,  # retry every 20mins
    on_backoff=backoff_logger
)
def send_gemini_request(images):
    contents = [
        """
        First, describe the first frame with all visible objects and their spatial positions relative to the viewer. After this, insert the marker 'end of description'. Next, describe dynamic changes and newly revealed objects or scenes in the following frames, always specifying their spatial positions relative to the first-frame viewpoint (e.g., behind the viewer, to the left, inside a container). Ensure the descriptions are chronologically ordered, in less than 6 sentences.
        """
    ]
    for image in images:
        contents.append(types.Part.from_bytes(data=image, mime_type='image/jpeg')) # type: ignore
    try:
        with genai.Client(api_key=api_key) as client:
            response = client.models.generate_content(
                model='gemini-2.5-flash',
                contents=contents # type: ignore
            )
    except Exception as ex:
        logger.error(f'catch exception: {ex}')
        raise ex
    return response.text


def compress_frame(img, target_width=None, target_height=None, jpeg_quality=80):
    """ resize and compress an image to reduce size.
    Args:
        img (np.ndarray): Input BGR image (from OpenCV).
        target_width (int or None): Desired width. If None, auto-scale from height.
        target_height (int or None): Desired height. If None, auto-scale from width.
        jpeg_quality (int): JPEG compression level (1–100, higher = better quality).
    Returns:
        str: Base64-encoded JPEG string.
    """
    h, w = img.shape[:2]
    # Compute resize ratio if both width/height not provided
    if target_width and target_height:
        new_w, new_h = target_width, target_height
    elif target_width:
        scale = target_width / w
        new_w, new_h = target_width, int(h * scale)
    elif target_height:
        scale = target_height / h
        new_w, new_h = int(w * scale), target_height
    else:
        new_w, new_h = w, h  # no resize
    # Resize only if dimensions change
    if (new_w, new_h) != (w, h):
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    # Encode to JPEG with given quality
    _, buffer = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
    return base64.b64encode(buffer).decode("utf-8")


def process_video_pyav(input_dir, filename, output_file):
    container = av.open(
        os.path.join(input_dir, filename),
        format='hevc' if filename.lower().endswith(".hevc") else None
    )
    stream = container.streams.video[0]
    start_frame_idx = 0
    end_frame_idx = SNIPPET_LENGTH
    base64_frames = []
    local_frame_idx = 0
    frame_cnter = 0
    for frame_idx, frame in enumerate(container.decode(stream)):
        frame_cnter += 1
        # read one frame each time
        assert frame_idx >= start_frame_idx and frame_idx <= end_frame_idx
        if local_frame_idx % FRMAE_GAP == 0:
            img = frame.to_ndarray(format='bgr24')
            _, buffer = cv2.imencode(".jpg", img)
            # base64_frames.append(base64.b64encode(buffer).decode("utf-8"))
            base64_frames.append(compress_frame(img, target_width=512, target_height=288))
        local_frame_idx += 1
        
        # hit the end
        if frame_idx == end_frame_idx:
            # (N + 1) images in total
            assert len(base64_frames) == SNIPPET_LENGTH / FRMAE_GAP + 1
            # request & response, write
            logger.info(f"start generating captions for {filename}, frame: {start_frame_idx} ~ {end_frame_idx}")
            caption = send_gemini_request(base64_frames)
            logger.info(f"finished generating captions for {filename}, frame: {start_frame_idx} ~ {end_frame_idx}")
            output_file.write(f"{_format(start_frame_idx)}-{_format(end_frame_idx)}:\n{caption}\n")

            # move pointers
            start_frame_idx = end_frame_idx
            end_frame_idx += SNIPPET_LENGTH
            base64_frames = [base64_frames[-1]]  # carry over last frame
            local_frame_idx = 1  # already has 1 frame
    container.close()
    logger.info(f"finished extracting frames from {filename}, which has {frame_cnter} frames")


def process_image_folder(folder_path, output_file):
    frames = sorted([p for p in Path(folder_path).glob("*") if p.suffix.lower() in image_exts])
    start_frame_idx = 0
    end_frame_idx = SNIPPET_LENGTH
    base64_frames = []
    local_frame_idx = 0
    frame_cnter = 0

    for frame_idx, frame_path in enumerate(frames):
        frame_cnter += 1
        assert frame_idx >= start_frame_idx and frame_idx <= end_frame_idx
        if local_frame_idx % FRMAE_GAP == 0:
            img = cv2.imread(str(frame_path))
            _, buffer = cv2.imencode(".jpg", img)
            base64_frames.append(base64.b64encode(buffer).decode("utf-8"))
        local_frame_idx += 1

        if frame_idx == end_frame_idx:
            assert len(base64_frames) == SNIPPET_LENGTH / FRMAE_GAP + 1
            logger.info(f"start generating captions for {folder_path}, frame: {start_frame_idx} ~ {end_frame_idx}")
            caption = send_gemini_request(base64_frames)
            logger.info(f"finished generating captions for {folder_path}, frame: {start_frame_idx} ~ {end_frame_idx}")
            output_file.write(f"{_format(start_frame_idx)}-{_format(end_frame_idx)}:\n{caption}\n")

            # move window
            start_frame_idx = end_frame_idx
            end_frame_idx += SNIPPET_LENGTH
            base64_frames = [base64_frames[-1]]
            local_frame_idx = 1
    
    logger.info(f"finished extracting frames from {folder_path}, which has {frame_cnter} frames")


if __name__ == "__main__":
    if not os.path.exists(OUTPUT_DIR):
        os.mkdir(OUTPUT_DIR)
    logger.info(f"Program starts, with snippet length = {SNIPPET_LENGTH}, frame gap = {FRMAE_GAP}")

    input_path = Path(INPUT_DIR)
    video_exts = {".mp4", ".hevc"}
    image_exts = {".png", ".jpg", ".jpeg"}
    # find all video/image files recursively
    filepaths = []
    for p in input_path.rglob("*"):
        if p.is_file() and p.suffix.lower() in video_exts:
            filepaths.append(p)
        elif p.is_dir():
            # only accept dirs with image files inside
            if any(q.suffix.lower() in image_exts for q in p.iterdir()):
                filepaths.append(p)


    print(f'len(filepaths) = {len(filepaths)}')
    for filepath in tqdm(filepaths):
        relative_path = filepath.relative_to(input_path)
        output_filename = f"{filepath.stem}.txt"
        output_path = Path(OUTPUT_DIR) / relative_path.parent / output_filename
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if output_path.exists():
            logger.warning(f"{output_path.name} exists, skip")
            continue

        with open(output_path, 'a+') as output_file:
            if filepath.is_file():  # video
                process_video_pyav(
                    input_dir=str(filepath.parent),
                    filename=filepath.name,
                    output_file=output_file
                )
            else:  # folder of frames
                process_image_folder(str(filepath), output_file)

