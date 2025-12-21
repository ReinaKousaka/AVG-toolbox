import os
import subprocess
from collections import defaultdict
from tqdm import tqdm


INPUT_LIST_FILE = "urls.txt"
OUTPUT_DIR = "251222-out-snippets"
TMP_VIDEO = "tmp.mp4"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---- Step 1: Load and sort ----
entries = []
with open(INPUT_LIST_FILE, "r") as f:
    for line in f:
        line = line.strip()
        if line.endswith(".json"):
            entries.append(line[:-5])   # strip .json
entries.sort()  # sort by name lexicographically

# ---- Step 2: Group by YouTube ID ----
groups = defaultdict(list)
for name in entries:
    # safely split from the right to extract last two numeric parts, because video name may contain underscores
    try:
        ytid, s, e = name.rsplit("_", 2)
    except ValueError:
        raise ValueError(f"Filename cannot be parsed using rsplit('_',2): '{name}'")

    # video name sanity check
    if not s.isdigit() or not e.isdigit():
        raise ValueError(f"Frame indices are not numeric in filename: '{name}'")
    start_f, end_f = int(s), int(e)
    if end_f <= start_f or start_f < 0:
        raise ValueError(f"Invalid frame range in filename: {name}")
    # store per YouTube ID
    groups[ytid].append((start_f, end_f))

# ---- Step 3: Handle each video ----
for ytid, segments in tqdm(groups.items()):
    url = f"https://www.youtube.com/watch?v={ytid}"
    print(f"\n=== Processing video: {url}  ===")

    # cleanup temp
    if os.path.exists(TMP_VIDEO):
        os.remove(TMP_VIDEO)

    # download video once
    # since pytube is unstable, use yt-dlp instead, see: https://github.com/yt-dlp/yt-dlp
    print(f"Downloading via yt-dlp for {len(segments)} segments...")
    subprocess.run([
        "yt-dlp",
        "-f", "mp4",
        "-o", TMP_VIDEO,
        url
    ], check=True)

    # ---- extract each segment ----
    for start_f, end_f in segments:
        print(f"Extracting frames {start_f} -> {end_f}")

        out_file = os.path.join(
            OUTPUT_DIR, f"{ytid}_{start_f}_{end_f}.mp4"
        )

        # -an: remove audo
        subprocess.run([
            "ffmpeg", "-y",
            "-i", TMP_VIDEO,
            "-vf", f"select='between(n\\,{start_f}\\,{end_f})',setpts=PTS-STARTPTS",
            "-an",
            "-c:v", "libx264",
            "-preset", "fast",
            out_file
        ], check=True)

        print(f"Saved snippet: {out_file}")

    # ---- cleanup video ----
    if os.path.exists(TMP_VIDEO):
        os.remove(TMP_VIDEO)

    print(f"Finished {url}, removed temp video")