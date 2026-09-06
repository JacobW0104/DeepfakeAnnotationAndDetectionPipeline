"""Annotate concepts with Qwen2-VL using region-cropped comparative prompts."""

import argparse
import csv
import os
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

from CNN_Classes_Transfer import decode_frames_uint8, probe_video

MODEL_ID = "Qwen/Qwen2-VL-2B-Instruct"
MASTER_CSV = "data/faceforensics_concept_dataset.csv"
OUT_CSV = "data/concept_labels_raw.csv"
REGION_DUMP_DIR = "region_check"

CSV_HEADERS = [
    "video_path",
    "reference_path",
    "method",
    "concept",
    "snippet",
    "p_real",
    "p_fake",
]

NUM_SNIPPETS = 3
FRAMES_PER_SNIPPET = 4
TIME_STEP = 0.1

CONCEPT_REGIONS = {
    "lip_sync": (0.34, 0.58, 0.68, 0.76),
    "blinking": (0.30, 0.34, 0.70, 0.51),
    "facial_boundary": (0.00, 0.00, 1.00, 1.00),
    "texture": (0.20, 0.46, 0.80, 0.60),
    "identity": (0.22, 0.28, 0.78, 0.82),
    "lighting": (0.00, 0.00, 1.00, 1.00),
}

QUESTIONS = {
    "lip_sync": (
        "Does Image 2 introduce lip-sync distortion, unnatural mouth shape or "
        "movement, or rendering artifacts in the mouth cavity compared to Image 1?"
    ),
    "blinking": (
        "Does Image 2 show abnormal eye shape, missing eyelid movement, or "
        "blinking artifacts relative to Image 1?"
    ),
    "facial_boundary": (
        "Does Image 2 show blur, colour bleeding, or blending seam artifacts "
        "where the face meets the skin, neck, or hair relative to Image 1?"
    ),
    "texture": (
        "Does Image 2 exhibit unnatural skin smoothing, artificial blurring, or "
        "texture flickering compared to Image 1?"
    ),
    "identity": (
        "Is the facial identity or structural geometry in Image 2 visibly "
        "altered from the subject in Image 1?"
    ),
    "lighting": (
        "Does Image 2 show inconsistent illumination, mismatched shadows, or "
        "unnatural specular highlights compared to Image 1?"
    ),
}

CONCEPT_LOWPASS = {"lighting": 32}

CONCEPTS = list(CONCEPT_REGIONS)

EXPECTED = {
    "neuraltextures": "lip_sync high, facial_boundary near zero",
    "faceswap": "facial_boundary high",
    "deepfakes": "facial_boundary high",
    "face2face": "lip_sync high, identity low",
}


def pair_snippet_indices(real_path, fake_path):
    """Matched frame indices for both videos of a pair."""
    n_real, fps = probe_video(real_path)
    n_fake, _ = probe_video(fake_path)
    n = min(n_real, n_fake)
    safe = max(0.5, n / fps - 0.5)
    return [
        [
            min(int((safe * frac + i * TIME_STEP) * fps), n - 1)
            for i in range(FRAMES_PER_SNIPPET)
        ]
        for frac in [0.20, 0.50, 0.80][:NUM_SNIPPETS]
    ]


def region_strip(frames, region, tile=224, lowpass=None):
    """Compose one region's frames into a single tiled image."""
    h, w = frames.shape[1:3]
    x0, y0, x1, y1 = region
    box = frames[:, int(y0 * h) : int(y1 * h), int(x0 * w) : int(x1 * w), :]
    tiles = [cv2.resize(b, (tile, tile)) for b in box]
    if lowpass:
        tiles = [
            cv2.resize(
                cv2.resize(t, (lowpass, lowpass), interpolation=cv2.INTER_AREA),
                (tile, tile),
                interpolation=cv2.INTER_LINEAR,
            )
            for t in tiles
        ]
    if len(tiles) == 4:
        grid = np.concatenate(
            [
                np.concatenate(tiles[:2], axis=1),
                np.concatenate(tiles[2:], axis=1),
            ],
            axis=0,
        )
    else:
        grid = np.concatenate(tiles, axis=1)
    return Image.fromarray(grid)


def build_messages(question):
    """Prompt asking one concept question about two images."""
    return [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "image"},
                {
                    "type": "text",
                    "text": (
                        "Image 1 is the authentic reference. Image 2 is the "
                        "candidate, showing the same region at the same moment.\n"
                        f"Question: {question}\n"
                        "Answer strictly with one word (Yes or No):"
                    ),
                },
            ],
        }
    ]


def load_model(device):
    """Load the annotator and resolve its Yes token ids."""
    import torch
    from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

    model = Qwen2VLForConditionalGeneration.from_pretrained(
        MODEL_ID,
        dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map="auto",
    )
    model.eval()
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    processor.tokenizer.padding_side = "left"

    yes_ids = []
    for t in (" Yes", "Yes", " yes", "yes"):
        ids = processor.tokenizer.encode(t, add_special_tokens=False)
        if len(ids) == 1:
            yes_ids.append(ids[0])
    if not yes_ids:
        raise RuntimeError("no single-token 'Yes' variant found in tokenizer")
    return model, processor, sorted(set(yes_ids))


def p_yes_batch(model, processor, yes_ids, items, device, batch_size=16):
    """P(Yes) for a batch of image pairs and questions."""
    import torch

    out = []
    for i in range(0, len(items), batch_size):
        chunk = items[i : i + batch_size]
        texts = [
            processor.apply_chat_template(
                build_messages(q), tokenize=False, add_generation_prompt=True
            )
            for _, _, q in chunk
        ]
        images = [[a, b] for a, b, _ in chunk]
        inputs = processor(
            text=texts, images=images, padding=True, return_tensors="pt"
        ).to(device)
        with torch.no_grad():
            logits = model(**inputs).logits[:, -1, :].float()
        probs = logits.softmax(dim=-1)[:, yes_ids].max(dim=-1).values
        out.extend(probs.tolist())
    return out


def already_done(path):
    """Video paths already annotated in the output CSV."""
    if not os.path.exists(path):
        return set()
    return set(pd.read_csv(path, usecols=["video_path"])["video_path"])


def append_rows(path, rows):
    """Append annotation rows to the output CSV."""
    exists = os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        if not exists:
            w.writerow(CSV_HEADERS)
        w.writerows(rows)


def load_pairs(limit=None):
    """Manipulated videos paired with their reference videos."""
    df = pd.read_csv(MASTER_CSV)
    df = df[df["video_path"] != df["reference_path"]]
    if limit:
        df = df.head(limit)
    return df[["video_path", "reference_path", "method"]].to_dict("records")


def dump_regions(n=6):
    """Write example region crops so the boxes can be inspected."""
    os.makedirs(REGION_DUMP_DIR, exist_ok=True)
    pairs = load_pairs(limit=n)
    for pair in tqdm(pairs, desc="regions"):
        snippets = pair_snippet_indices(pair["reference_path"], pair["video_path"])
        frames = decode_frames_uint8(
            pair["video_path"], snippets[1], detect=True
        )
        stem = Path(pair["video_path"]).stem
        Image.fromarray(frames[0]).save(f"{REGION_DUMP_DIR}/{stem}_00_full.png")
        for concept, region in CONCEPT_REGIONS.items():
            region_strip(
                frames[:1], region, lowpass=CONCEPT_LOWPASS.get(concept)
            ).save(
                f"{REGION_DUMP_DIR}/{stem}_{concept}.png"
            )
    print(f"\nwrote {REGION_DUMP_DIR}/ -- open these and check that:")
    print("  *_lip_sync.png    shows a mouth, not a chin or neck")
    print("  *_blinking.png    shows both eyes")
    print("  *_texture.png     shows cheeks, not eyes or mouth")
    print("If they are consistently off, use a real face detector before")
    print("relabelling -- the crop is a fixed centre crop, not detection.")


def audit():
    """Print mean concept scores by manipulation method."""
    long = pd.read_csv(OUT_CSV)
    long["delta"] = long["p_fake"] - long["p_real"]
    wide = (
        long.groupby(["video_path", "method", "concept"])["delta"]
        .mean()
        .unstack()
        .reset_index()
    )
    print(f"{len(wide)} videos labelled\n")
    print("mean raw delta by method:")
    tab = wide.groupby("method")[CONCEPTS].mean()
    print(tab.round(4).to_string())
    print("\nper-method ranking of concepts (1 = strongest):")
    print(tab.rank(axis=1, ascending=False).astype(int).to_string())
    print("\ninter-concept correlation (want LOW; old labels were 0.30-0.70):")
    corr = wide[CONCEPTS].corr()
    print(corr.round(2).to_string())
    off = corr.values[~np.eye(len(CONCEPTS), dtype=bool)]
    print(f"\nmean off-diagonal correlation: {off.mean():.3f}")
    print("\nexpected signatures:")
    for m, e in EXPECTED.items():
        print(f"  {m:<16} {e}")
    print("\nIf methods still rank concepts identically and correlations stay")
    print("high, the region-crop hypothesis is wrong -- stop and rethink.")


def run(args):
    """Annotate every pair and write raw probabilities."""
    import time as _time

    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"loading {MODEL_ID} on {device}")
    model, processor, yes_ids = load_model(device)
    print(f"'Yes' token ids: {yes_ids}")

    pairs = load_pairs(args.limit)
    done = already_done(OUT_CSV)
    pairs = [p for p in pairs if p["video_path"] not in done]
    if done:
        print(f"resuming: {len(done)} already labelled, {len(pairs)} to go")

    real_memo = {}
    t_decode = t_vlm = 0.0
    n_passes = 0

    for pair in tqdm(pairs, desc="relabel", unit="video"):
        try:
            t0 = _time.time()
            snippets = pair_snippet_indices(
                pair["reference_path"], pair["video_path"]
            )
            flat = [i for s in snippets for i in s]
            real_f = decode_frames_uint8(
                pair["reference_path"], flat, detect=True
            )
            fake_f = decode_frames_uint8(
                pair["video_path"], flat, detect=True
            )
            t_decode += _time.time() - t0

            items, keys = [], []
            for s_idx, s in enumerate(snippets):
                lo, hi = s_idx * FRAMES_PER_SNIPPET, (s_idx + 1) * FRAMES_PER_SNIPPET
                r4, f4 = real_f[lo:hi], fake_f[lo:hi]
                for concept in CONCEPTS:
                    region = CONCEPT_REGIONS[concept]
                    lp = CONCEPT_LOWPASS.get(concept)
                    r_strip = region_strip(r4, region, lowpass=lp)
                    f_strip = region_strip(f4, region, lowpass=lp)
                    memo_key = (pair["reference_path"], tuple(s), concept)
                    if memo_key not in real_memo:
                        items.append((r_strip, r_strip, QUESTIONS[concept]))
                        keys.append(("real", memo_key))
                    items.append((r_strip, f_strip, QUESTIONS[concept]))
                    keys.append(("fake", (s_idx, concept)))

            t0 = _time.time()
            probs = p_yes_batch(
                model, processor, yes_ids, items, device, args.batch_size
            )
            t_vlm += _time.time() - t0
            n_passes += len(items)

            fake_p = {}
            for (kind, key), p in zip(keys, probs):
                if kind == "real":
                    real_memo[key] = p
                else:
                    fake_p[key] = p

            rows = []
            for s_idx, s in enumerate(snippets):
                for concept in CONCEPTS:
                    rows.append(
                        [
                            pair["video_path"],
                            pair["reference_path"],
                            pair["method"],
                            concept,
                            s_idx,
                            real_memo[(pair["reference_path"], tuple(s), concept)],
                            fake_p[(s_idx, concept)],
                        ]
                    )
            append_rows(OUT_CSV, rows)

        except Exception as exc:
            print(f"\n[ERROR] {pair['video_path']}: {exc!r}")

    if args.time and pairs:
        n = len(pairs)
        print(f"\n--- timing over {n} videos ---")
        print(f"decode : {t_decode:7.1f}s  ({t_decode / n:.3f} s/video)")
        print(f"VLM    : {t_vlm:7.1f}s  ({t_vlm / n:.3f} s/video, "
              f"{n_passes / max(t_vlm, 1e-9):.1f} passes/s)")
        per = (t_decode + t_vlm) / n
        print(f"total  : {per:.3f} s/video -> 5000 videos = "
              f"{per * 5000 / 3600:.2f} hours")
        print(f"real-side memo hit rate: "
              f"{1 - len(real_memo) / max(n * NUM_SNIPPETS * len(CONCEPTS), 1):.1%}")


def main():
    """Parse arguments and dispatch to the requested action."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--dump-regions", action="store_true")
    ap.add_argument("--audit", action="store_true")
    ap.add_argument("--time", action="store_true")
    args = ap.parse_args()

    if args.dump_regions:
        dump_regions()
    elif args.audit:
        audit()
    else:
        run(args)


if __name__ == "__main__":
    main()
