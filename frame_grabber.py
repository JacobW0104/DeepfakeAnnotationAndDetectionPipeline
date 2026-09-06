from moviepy import VideoFileClip
from pathlib import Path
import os
import cv2
import numpy as np
from PIL import Image

def frame_grabber(filepath:str, start_time:int, time_step:float, frame_count=4, name=None):
    """Save frames taken at the given times from a video."""
    output_dir = "single_frames"
    os.makedirs(output_dir, exist_ok=True)

    if name is None:
        name = Path(filepath).stem

    video = VideoFileClip(filepath)
    saved_paths = []

    timestamps = [start_time+time_step*i for i in range(frame_count)]

    for t in timestamps:
        output_path = os.path.join(output_dir, f"{name}_at_{t:.2f}s.png")
        video.save_frame(output_path, t)
        saved_paths.append(output_path)

    video.close()

    return saved_paths


def create_frame_strip(frames: list, target_size=(256, 256)) -> Image.Image:
    """Compose saved frames into a single strip image."""
    processed = []
    for f in frames:
        img = cv2.imread(f) if isinstance(f, str) else f

        face_img = crop_face(img, padding=0.25)

        face_resized = cv2.resize(face_img, target_size)
        processed.append(face_resized)

    strip_bgr = cv2.hconcat(processed)
    strip_rgb = cv2.cvtColor(strip_bgr, cv2.COLOR_BGR2RGB)

    Image.fromarray(strip_rgb).save('single_frames/test.jpg')

    return Image.fromarray(strip_rgb)


def crop_face(frame: np.ndarray, padding: float = 0.3) -> np.ndarray:
    """Crop a frame to a padded central face region."""
    h_img, w_img = frame.shape[:2]

    if hasattr(cv2, 'CascadeClassifier'):
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        face_cascade = cv2.CascadeClassifier(cascade_path)

        if not face_cascade.empty():
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))

            if len(faces) > 0:
                x, y, w, h = max(faces, key=lambda rect: rect[2] * rect[3])

                pad_w = int(w * padding)
                pad_h = int(h * padding)

                x1 = max(0, x - pad_w)
                y1 = max(0, y - pad_h)
                x2 = min(w_img, x + w + pad_w)
                y2 = min(h_img, y + h + pad_h)

                return frame[y1:y2, x1:x2]

    cx = w_img // 2
    cw = int(w_img * 0.55)
    ch = int(h_img * 0.60)

    x1 = max(0, cx - cw // 2)
    x2 = min(w_img, cx + cw // 2)
    y1 = int(h_img * 0.05)
    y2 = min(h_img, y1 + ch)

    return frame[y1:y2, x1:x2]

def clean_frames_file():
    """Delete temporary frame files."""
    for file_path in Path('single_frames').iterdir():
        if file_path.is_file():
            file_path.unlink()
