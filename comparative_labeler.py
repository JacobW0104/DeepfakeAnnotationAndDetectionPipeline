import csv
import gc
import os
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
from qwen_vl_utils import process_vision_info

from frame_grabber import clean_frames_file, create_frame_strip, frame_grabber

MODEL_ID = "Qwen/Qwen2-VL-2B-Instruct"
CSV_OUTPUT_PATH = "data/faceforensics_concept_dataset.csv"

CSV_HEADERS = [
    "video_path",
    "reference_path",
    "method",
    "is_fake",
    "c_lip_sync",
    "c_blinking",
    "c_facial_boundary",
    "c_texture",
    "c_identity",
    "c_lighting",
]

FORENSIC_QUESTIONS = {
    "lip_sync_mismatch": (
        "Focus on the mouth, lips, and teeth. Does Image 2 introduce lip-sync distortion, "
        "unnatural mouth shape or movement, or subtle rendering artifacts in the mouth cavity compared to Image 1?"
    ),
    "blinking_anomaly": (
        "Focus strictly on the eye and eyelid region across the 4 frames. "
        "Does Image 2 show abnormal eye shape, missing eyelid movement, or blinking twitching relative to Image 1?"
    ),
    "facial_boundary_drift": (
        "Focus on the outer boundary of the face (jawline, chin, forehead edges, and side cheeks). "
        "Does Image 2 show subtle blur, color bleeding, or blending seam artifacts where the face meets the skin, neck, or hair relative to Image 1?"
    ),
    "texture_flickering": (
        "Inspect the fine skin texture across the cheeks, forehead, and chin. "
        "Does Image 2 exhibit unnatural skin smoothing, artificial blurring, or texture flickering compared to Image 1?"
    ),
    "identity_inconsistency": (
        "Ignore hair, clothing, neck, and background entirely. Focus exclusively on the inner facial bone structure, nose shape, and jaw geometry. "
        "Is the facial identity or structural geometry in Image 2 visibly altered from the subject in Image 1?"
    ),
    "lighting_shadow_discrepancy": (
        "Focus on facial highlights, nose shadows, and cheek illumination. "
        "Is there an inconsistent shadow direction, harsh artificial glow, or lighting mismatch on the face in Image 2 relative to Image 1?"
    ),
}

def scan_faceforensics_dataset(dataset_root: str | Path) -> list[dict]:
    """List every video with its reference, method and authenticity label."""
    dataset_root = Path(dataset_root)
    original_dir = dataset_root / "original"

    methods = [
        "Deepfakes",
        "Face2Face",
        "FaceShifter",
        "FaceSwap",
        "NeuralTextures",
    ]

    if not original_dir.exists():
        raise FileNotFoundError(
            f"Original directory not found at {original_dir}"
        )

    pairs = []
    original_videos = sorted(list(original_dir.glob("*.mp4")))

    print(
        f"Found {len(original_videos)} pristine original videos in {original_dir}"
    )

    for real_path in original_videos:
        video_id = real_path.stem

        pairs.append({
            "video_path": str(real_path),
            "reference_path": str(real_path),
            "method": "original",
        })

        for method in methods:
            method_dir = dataset_root / method
            if not method_dir.exists():
                method_dir = dataset_root / method.lower()
                if not method_dir.exists():
                    continue

            matching_fakes = sorted(list(method_dir.glob(f"{video_id}_*.mp4")))

            if not matching_fakes:
                matching_fakes = sorted(
                    list(method_dir.glob(f"{video_id}.mp4"))
                )

            for fake_path in matching_fakes:
                pairs.append({
                    "video_path": str(fake_path),
                    "reference_path": str(real_path),
                    "method": method.lower(),
                })

    print(f"Total video pairs indexed for processing: {len(pairs)}")
    return pairs


def get_processed_video_paths(csv_path: str) -> set[str]:
    """Video paths already present in the output CSV."""
    if not os.path.exists(csv_path):
        return set()
    try:
        df = pd.read_csv(csv_path)
        if "video_path" in df.columns:
            return set(df["video_path"].astype(str).tolist())
    except Exception as e:
        print(f"Warning reading CSV for resuming: {e}")
    return set()


def append_to_csv( csv_path: str, video_path: str, reference_path: str, method: str, concept_vector: list[float]):
    """Append one video's concept vector to the output CSV."""
    file_exists = os.path.exists(csv_path)
    is_fake = 0 if method.lower() in ["original", "real"] else 1

    row = [video_path, reference_path, method, is_fake] + [
        round(v, 4) for v in concept_vector
    ]

    with open(csv_path, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(CSV_HEADERS)
        writer.writerow(row)


def get_video_duration(video_path: str) -> float:
    """Duration of a video in seconds."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return 3.0
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    cap.release()
    if fps > 0 and frame_count > 0:
        return frame_count / fps
    return 3.0

def extract_raw_yes_prob(model, processor, tokenizer, device, real_frames: list, target_frames: list,
                         concept_question: str) -> float:
    """P(Yes) for one question about a pair of frame strips."""
    real_strip = create_frame_strip(real_frames)
    target_strip = create_frame_strip(target_frames)

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": real_strip},
                {"type": "image", "image": target_strip},
                {
                    "type": "text",
                    "text": (
                        "Image 1 shows 4 sequential cropped facial frames from a reference pristine video clip.\n"
                        "Image 2 shows 4 sequential cropped facial frames from a target video clip that may be identical, authentic, or manipulated.\n\n"
                        "Task: Compare Image 2 directly against Image 1.\n"
                        "If Image 2 is identical to Image 1 or shows no introduced artifacts, your answer must be No.\n"
                        f"Question: {concept_question}\n"
                        "Answer strictly with one word (Yes or No):"
                    ),
                },
            ],
        }
    ]

    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model(**inputs)
        next_token_logits = outputs.logits[0, -1, :]

        yes_tokens = [
            tokenizer.encode(t, add_special_tokens=False)[0]
            for t in [" Yes", "Yes", " yes"]
            if len(tokenizer.encode(t, add_special_tokens=False)) > 0
        ]
        no_tokens = [
            tokenizer.encode(t, add_special_tokens=False)[0]
            for t in [" No", "No", " no"]
            if len(tokenizer.encode(t, add_special_tokens=False)) > 0
        ]

        max_yes_logit = max(
            [next_token_logits[tid].item() for tid in yes_tokens]
        )
        max_no_logit = max(
            [next_token_logits[tid].item() for tid in no_tokens]
        )

        selected_logits = torch.tensor([max_no_logit, max_yes_logit])
        probs = F.softmax(selected_logits, dim=0)
        return probs[1].item()


def calculate_concept_probability( model, processor, tokenizer, device, real_frame_path: str, fake_frame_path: str,
                                   concept_question: str, num_snippets: int = 3, time_step: float = 0.10,
                                   signal_scale: float = 5.0) -> float:
    """Mean P(Yes) difference for one concept across a video."""
    real_dur = get_video_duration(real_frame_path)
    fake_dur = get_video_duration(fake_frame_path)

    safe_duration = max(0.5, min(real_dur, fake_dur) - 0.5)
    start_times = [safe_duration * frac for frac in [0.20, 0.50, 0.80]][
        :num_snippets
    ]

    deltas = []
    for s_idx, start_t in enumerate(start_times):
        real_frames = frame_grabber(
            real_frame_path,
            start_time=start_t,
            time_step=time_step,
            name=f"Real_s{s_idx}",
        )
        fake_frames = frame_grabber(
            fake_frame_path,
            start_time=start_t,
            time_step=time_step,
            name=f"Fake_s{s_idx}",
        )

        p_real = extract_raw_yes_prob(
            model,
            processor,
            tokenizer,
            device,
            real_frames,
            real_frames,
            concept_question,
        )
        p_fake = extract_raw_yes_prob(
            model,
            processor,
            tokenizer,
            device,
            real_frames,
            fake_frames,
            concept_question,
        )

        delta = max(0.0, p_fake - p_real)
        deltas.append(delta)

        clean_frames_file()

    avg_delta = float(np.mean(deltas)) if deltas else 0.0

    if avg_delta < 0.008:
        return 0.0000

    scaled_signal = min(1.0, avg_delta * signal_scale)
    return round(scaled_signal, 4)

def main():
    """Annotate every video and write the master dataset CSV."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Initializing Qwen2-VL Model on device: {device}")

    model = Qwen2VLForConditionalGeneration.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map="auto",
    )
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    tokenizer = processor.tokenizer

    dataset_root = "FaceForensics"
    video_pairs = scan_faceforensics_dataset(dataset_root)

    processed_set = get_processed_video_paths(CSV_OUTPUT_PATH)
    if processed_set:
        print(
            f"Resuming session: {len(processed_set)} videos already completed in {CSV_OUTPUT_PATH}."
        )

    unprocessed_pairs = [
        p for p in video_pairs if p["video_path"] not in processed_set
    ]
    print(
        f"Ready to process remaining {len(unprocessed_pairs)} video pairs."
    )

    pbar = tqdm(unprocessed_pairs, desc="Processing Videos", unit="video")
    for pair in pbar:
        target_path = pair["video_path"]
        ref_path = pair["reference_path"]
        method = pair["method"]

        pbar.set_postfix({"current": Path(target_path).name, "method": method})

        try:
            concept_vector = []
            for concept_name, question in FORENSIC_QUESTIONS.items():
                prob = calculate_concept_probability(
                    model=model,
                    processor=processor,
                    tokenizer=tokenizer,
                    device=device,
                    real_frame_path=ref_path,
                    fake_frame_path=target_path,
                    concept_question=question,
                )
                concept_vector.append(prob)

            append_to_csv(
                csv_path=CSV_OUTPUT_PATH,
                video_path=target_path,
                reference_path=ref_path,
                method=method,
                concept_vector=concept_vector,
            )

        except Exception as e:
            print(
                f"\n[ERROR] Error processing {target_path}: {e}. Skipping..."
            )

        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

    print(
        f"\nprocessing complete! Final dataset saved to: {CSV_OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()
