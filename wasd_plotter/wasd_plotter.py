import cv2
import json


ROT_MAP = {"UP": "^", "DOWN": "v", "LEFT": "<", "RIGHT": ">"}


def draw_key(frame, label, x, y, pressed):
    size = 45
    if pressed:
        color = (0, 255, 0)
        thickness = -1
        text_color = (0, 0, 0)
    else:
        color = (200, 200, 200)
        thickness = 2
        text_color = (255, 255, 255)

    cv2.rectangle(frame, (x, y), (x + size, y + size), color, thickness)

    cv2.putText(
        frame, label, (x + 13, y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, text_color, 2
    )


def draw_keyboard(frame, pressed):

    h, w, _ = frame.shape

    base_x = 50
    base_y = h - 150

    # WASD
    draw_key(frame, "W", base_x + 45, base_y, "W" in pressed)
    draw_key(frame, "A", base_x, base_y + 45, "A" in pressed)
    draw_key(frame, "S", base_x + 45, base_y + 45, "S" in pressed)
    draw_key(frame, "D", base_x + 90, base_y + 45, "D" in pressed)

    # Arrows
    ax = w - 200
    ay = base_y

    draw_key(frame, "^", ax + 45, ay, "^" in pressed)
    draw_key(frame, "<", ax, ay + 45, "<" in pressed)
    draw_key(frame, "v", ax + 45, ay + 45, "v" in pressed)
    draw_key(frame, ">", ax + 90, ay + 45, ">" in pressed)


def load_key_frames(json_path):

    with open(json_path) as f:
        data = json.load(f)

    frame_map = {}

    for item in data:

        pressed = []

        move = item["movement"]
        rot = item["rotation"]

        if move != "nothing":
            pressed.append(move)

        if rot != "nothing":
            pressed.append(ROT_MAP[rot])

        frame_map[item["frame_idx"]] = pressed

    return frame_map


def overlay(video_in, video_out, json_file):

    frame_map = load_key_frames(json_file)

    cap = cv2.VideoCapture(video_in)

    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out = cv2.VideoWriter(video_out, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    frame_id = 0

    while True:

        ret, frame = cap.read()
        if not ret:
            break

        pressed = frame_map.get(frame_id, [])

        draw_keyboard(frame, pressed)

        out.write(frame)

        frame_id += 1

    cap.release()
    out.release()


overlay("demo2.mp4", "output.mp4", "keys.json")
