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
from multiprocessing import Process
from pathlib import Path
from tqdm import tqdm
import logging
import backoff
import cv2
import av  # for .hevc
import json, re
from google import genai
from google.genai import types


# FILL THIS CONFIDENTIAL KEY! Better to use env variable by:
# GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
VIDEO_EXTS = {".mp4", ".hevc"}
IMAGE_EXTS = {".png", ".jpg", ".jpeg"}
LOG_DIR = Path("gemini_log")
LOG_FORMAT = "[%(asctime)s] - %(levelname)s: %(message)s"
LOG_DATEFMT = "%H:%M:%S"

parser = argparse.ArgumentParser()
parser.add_argument(
    "--snippet_length",
    type=int,
    default=96,
    help="snippet length of subvideo to caption",
)
parser.add_argument("--frame_gap", type=int, default=16, help="frame gap to feed")
parser.add_argument("--video_dir", type=str, default="vids", help="input path of video")
parser.add_argument(
    "--output_dir", type=str, default="caption-gemini-out", help="output path"
)
parser.add_argument(
    "--gemini_api_keys",
    nargs="+",
    default=None,
    help="one or more Gemini API keys to parallelize requests",
)

args = parser.parse_args()
SNIPPET_LENGTH = args.snippet_length
FRMAE_GAP = args.frame_gap
INPUT_DIR = args.video_dir
OUTPUT_DIR = args.output_dir
GEMINI_API_KEYS = args.gemini_api_keys or ([GEMINI_API_KEY] if GEMINI_API_KEY else None)
if not GEMINI_API_KEYS:
    raise ValueError(
        "Gemini API key not provided. Set GEMINI_API_KEY env variable or pass --gemini_api_keys."
    )
assert SNIPPET_LENGTH % FRMAE_GAP == 0


logging.basicConfig(
    level=logging.INFO,  # mute Google's DEBUG level messages
    format=LOG_FORMAT,
    datefmt=LOG_DATEFMT,
)
logger = logging.getLogger(__name__)


def _log_filename_prefix(api_key: str) -> str:
    prefix = api_key[-5:] if api_key else "empty"
    return re.sub(r"[^A-Za-z0-9_-]", "_", prefix)


def get_log_file_for_key(api_key: str) -> Path:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    return LOG_DIR / f"{_log_filename_prefix(api_key)}.log"


def configure_logger_for_api_key(api_key: str):
    log_file = get_log_file_for_key(api_key)
    for handler in logger.handlers:
        if isinstance(handler, logging.FileHandler) and getattr(
            handler, "baseFilename", ""
        ) == str(log_file):
            return log_file
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(logging.Formatter(LOG_FORMAT, LOG_DATEFMT))
    logger.addHandler(file_handler)
    return log_file


def _format(number: int) -> str:
    return f"frame_{str(number).zfill(10)}"


def backoff_logger(details):
    """logger for backoff"""
    tries = details["tries"]
    if tries < 5:
        logger.info(f"Retry {tries} times")
    else:
        logger.warning(f"Retry {tries} times")


# wrapper to retry on failures
@backoff.on_exception(
    backoff.constant,
    Exception,
    interval=60 * 20,  # retry every 20mins
    on_backoff=backoff_logger,
)
def send_gemini_request(
    images,
    contents=[
        """
        You are a precise and concise first-person video scene narrator.

        **Detailed Caption Instructions:**
        [first]: Describe the first frame with all visible objects and their static spatial positions (e.g., "a box on the left, a bottle on the right").
        [adjacent_list]: For **every adjacent pair of frames** (Frame N and Frame N+1), describe dynamic changes, newly revealed objects, or subtle object movements, always specifying their static spatial positions (e.g., "the object on the left is moving," "a new chair appears in the upper right"). The descriptions must be chronologically ordered.
        [overall]: Provide a single, detailed, and chronologically ordered summary of the entire video sequence, focusing on key actions and object relations.

        **Brief Caption Instructions:**
        [brief]: Generate a single, concise, and compelling caption (5-8 words) summarizing the main scene and action.

        Ensure all descriptions are accurate, information-rich, and the [overall] caption is less than 6 sentences.

        Return the descriptions in the format below: {"detailed": {"first": "...", "adjacent_list": ["...", "..."], "overall": "..."}, "brief": "..."}. Respond **only** with a valid JSON object that can be parsed by Python's `json.loads`. Do not format into Markdown code blocks. Your response must start with `{` and end with `}`. Do not include any NSFW content or swear words in your descriptions. Do not include any backslash symbol in your response.
        """
    ],
):

    for image in images:
        contents.append(types.Part.from_bytes(data=image, mime_type="image/jpeg"))  # type: ignore
    try:
        client = genai.Client(api_key=GEMINI_API_KEY)
        response = client.models.generate_content(
            model="gemini-2.5-flash", contents=contents  # type: ignore
        )
    except Exception as ex:
        logger.error(f"catch exception: {ex}")
        raise ex
    text = response.text
    caption_simple = client.models.generate_content(  # type: ignore
        model="gemini-2.5-flash",
        contents=[
            f"Summarize the description of frames into a concise caption of no more than 20 words, without any format. the description is: [descriptions start] {text} [descriptions end], ignoring the format such as first or remaining and json structure in the description",
        ],
    ).text
    return response.text, caption_simple


def compress_frame(img, target_width=None, target_height=None, jpeg_quality=80):
    """resize and compress an image to reduce size.
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
        format="hevc" if filename.lower().endswith(".hevc") else None,
    )
    stream = container.streams.video[0]
    start_frame_idx = 0
    end_frame_idx = SNIPPET_LENGTH
    base64_frames = []
    local_frame_idx = 0
    frame_cnter = 0
    jsons = {}
    if os.path.exists(os.path.join(OUTPUT_DIR, f"{filename.split('.')[0]}.json")):
        logger.warning(
            f"{filename.split('.')[0]}.json exists, skip processing {filename}"
        )
        return
    for frame_idx, frame in enumerate(container.decode(stream)):
        frame_cnter += 1
        # read one frame each time
        assert frame_idx >= start_frame_idx and frame_idx <= end_frame_idx
        if local_frame_idx % FRMAE_GAP == 0:
            img = frame.to_ndarray(format="bgr24")
            _, buffer = cv2.imencode(".jpg", img)
            # base64_frames.append(base64.b64encode(buffer).decode("utf-8"))
            base64_frames.append(
                compress_frame(img, target_width=400, target_height=216)
            )
        local_frame_idx += 1

        # hit the end
        if frame_idx == end_frame_idx:
            # (N + 1) images in total
            assert len(base64_frames) == SNIPPET_LENGTH / FRMAE_GAP + 1
            # request & response, write
            logger.info(
                f"start generating captions for {filename}, frame: {start_frame_idx} ~ {end_frame_idx}"
            )
            caption, caption_simple = send_gemini_request(base64_frames)
            logger.info(
                f"finished generating captions for {filename}, frame: {start_frame_idx} ~ {end_frame_idx}"
            )
            output_file.write(
                f"{_format(start_frame_idx)}-{_format(end_frame_idx)}:\n{caption}\n"
            )
            try:

                caption = json.loads(caption)
            except json.JSONDecodeError:
                # try to fix common issues using regex
                caption = re.sub(r",\s*}", "}", caption)  # remove trailing commas
                caption = re.sub(r",\s*]", "]", caption)  # remove trailing commas
                caption = re.sub(r"(\w+):", r'"\1":', caption)  # quote keys
                caption = re.sub(r"'", '"', caption)  # convert single to double quotes
                try:
                    caption = json.loads(caption)
                except json.JSONDecodeError:
                    logger.error(
                        f"Failed to parse JSON after fixes for frames {start_frame_idx}-{end_frame_idx} in {filename}. Raw response: {caption}"
                    )
                    caption = caption
            if not isinstance(caption, dict) and isinstance(caption, str):
                caption = caption.split("remaining")
                caption = {
                    "first": caption[0][10:].replace('"', "").replace(":", ""),
                    "remaining": caption[1][1:]
                    .replace('"', "")
                    .replace(":", "")
                    .replace("}", ""),
                }
            assert isinstance(caption, dict)
            caption["short_caption"] = caption_simple
            jsons[f"{_format(start_frame_idx)}-{_format(end_frame_idx)}"] = caption
            # move pointers
            start_frame_idx = end_frame_idx
            end_frame_idx += SNIPPET_LENGTH
            base64_frames = [base64_frames[-1]]  # carry over last frame
            local_frame_idx = 1  # already has 1 frame
    with open(os.path.join(OUTPUT_DIR, f"{filename.split('.')[0]}.json"), "w") as f:
        json.dump(jsons, f, indent=4)
    container.close()
    logger.info(
        f"finished extracting frames from {filename}, which has {frame_cnter} frames"
    )


def process_image_folder(folder_path, output_file):
    frames = sorted(
        [p for p in Path(folder_path).glob("*") if p.suffix.lower() in IMAGE_EXTS]
    )
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
            logger.info(
                f"start generating captions for {folder_path}, frame: {start_frame_idx} ~ {end_frame_idx}"
            )
            caption = send_gemini_request(base64_frames)
            logger.info(
                f"finished generating captions for {folder_path}, frame: {start_frame_idx} ~ {end_frame_idx}"
            )
            output_file.write(
                f"{_format(start_frame_idx)}-{_format(end_frame_idx)}:\n{caption}\n"
            )

            # move window
            start_frame_idx = end_frame_idx
            end_frame_idx += SNIPPET_LENGTH
            base64_frames = [base64_frames[-1]]
            local_frame_idx = 1

    logger.info(
        f"finished extracting frames from {folder_path}, which has {frame_cnter} frames"
    )


def chunk_filepaths(filepaths, num_chunks):
    if num_chunks <= 0:
        raise ValueError("num_chunks must be positive")
    if not filepaths:
        return [[] for _ in range(num_chunks)]
    base, remainder = divmod(len(filepaths), num_chunks)
    chunks = []
    start = 0
    for idx in range(num_chunks):
        extra = 1 if idx < remainder else 0
        end = start + base + extra
        chunks.append(filepaths[start:end])
        start = end
    return chunks


def process_path_batch(filepaths, input_path, show_progress=False):
    iterator = tqdm(filepaths) if show_progress else filepaths
    for filepath in iterator:
        relative_path = filepath.relative_to(input_path)
        output_filename = f"{filepath.stem}.txt"
        output_path = Path(OUTPUT_DIR) / relative_path.parent / output_filename
        output_path.parent.mkdir(parents=True, exist_ok=True)
        # if output_path.exists():
        #     logger.warning(f"{output_path.name} exists, skip")
        #     continue

        with open(output_path, "a+") as output_file:
            if filepath.is_file():  # video
                process_video_pyav(
                    input_dir=str(filepath.parent),
                    filename=filepath.name,
                    output_file=output_file,
                )
            else:
                process_image_folder(folder_path=str(filepath), output_file=output_file)


def worker_process(filepaths, api_key, input_path):
    global GEMINI_API_KEY
    GEMINI_API_KEY = api_key
    configure_logger_for_api_key(api_key)
    logger.info(
        f"Worker handling {len(filepaths)} paths with key ending {api_key[-4:]}"
    )
    process_path_batch(filepaths, input_path, show_progress=False)


if __name__ == "__main__":
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    logger.info(
        f"Program starts, with snippet length = {SNIPPET_LENGTH}, frame gap = {FRMAE_GAP}"
    )

    input_path = Path(INPUT_DIR)
    # find all video/image files recursively
    filepaths = []
    for p in input_path.rglob("*"):
        if p.is_file() and p.suffix.lower() in VIDEO_EXTS:
            filepaths.append(p)
        elif p.is_dir():
            # only accept dirs with image files inside
            if any(q.suffix.lower() in IMAGE_EXTS for q in p.iterdir()):
                filepaths.append(p)
    filepaths = sorted(filepaths)
    print(f"len(filepaths) = {len(filepaths)}")
    for api_key in GEMINI_API_KEYS:
        log_file = get_log_file_for_key(api_key)
        log_file.touch(exist_ok=True)
    if not filepaths:
        logger.warning("No supported video or image folders found to process.")
    else:
        if len(GEMINI_API_KEYS) == 1:
            GEMINI_API_KEY = GEMINI_API_KEYS[0]
            configure_logger_for_api_key(GEMINI_API_KEY)
            process_path_batch(filepaths, input_path, show_progress=True)
        else:
            chunks = chunk_filepaths(filepaths, len(GEMINI_API_KEYS))
            processes = []
            for api_key, chunk in zip(GEMINI_API_KEYS, chunks):
                if not chunk:
                    log_file = get_log_file_for_key(api_key)
                    with open(log_file, "a") as lf:
                        lf.write("No filepaths assigned to this API key.\n")
                    logger.info(
                        f"No filepaths assigned to API key ending {api_key[-4:]}"
                    )
                    continue
                proc = Process(
                    target=worker_process,
                    args=(chunk, api_key, input_path),
                )
                proc.start()
                processes.append(proc)
            for proc in processes:
                proc.join()
