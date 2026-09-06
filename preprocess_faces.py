import argparse
import os
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
from tqdm import tqdm

from CNN_Classes_Transfer import cache_path_for, decode_clip_uint8

MASTER_CSV = "data/faceforensics_concept_dataset.csv"


def build_one(args):
    """Cache one video's crops."""
    video_path, root, detect, img_size = args
    out = cache_path_for(video_path, root)
    if out.exists():
        return video_path, "skip", None
    try:
        clip = decode_clip_uint8(
            video_path, img_size=(img_size, img_size), detect=detect
        )
    except Exception as exc:
        return video_path, "error", repr(exc)

    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".tmp.npy")
    np.save(tmp, clip)
    os.replace(tmp, out)
    return video_path, "ok", None


def verify_one(args):
    """Re-decode one video and compare it against the cache."""
    video_path, root, detect, img_size = args
    out = cache_path_for(video_path, root)
    if not out.exists():
        return video_path, "missing", None
    cached = np.load(out)
    live = decode_clip_uint8(
        video_path, img_size=(img_size, img_size), detect=detect
    )
    if cached.shape != live.shape or not np.array_equal(cached, live):
        return video_path, "MISMATCH", None
    return video_path, "ok", None


def run(fn, paths, workers, desc):
    """Run a worker function over all videos in a process pool."""
    counts = {}
    errors = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(fn, p) for p in paths]
        for fut in tqdm(as_completed(futures), total=len(futures), desc=desc):
            path, status, detail = fut.result()
            counts[status] = counts.get(status, 0) + 1
            if status not in ("ok", "skip"):
                errors.append((path, status, detail))
    return counts, errors


def main():
    """Build or verify the crop cache."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=None,
                    help="only process the first N videos (for a trial run)")
    ap.add_argument("--verify", action="store_true",
                    help="re-decode and compare against the cache")
    ap.add_argument("--detect", action="store_true",
                    help="crop to a YuNet-detected face instead of the centre "
                         "crop; falls back to the centre crop when no face is "
                         "found. Write these to a SEPARATE --cache-root.")
    ap.add_argument("--cache-root", default=None,
                    help="defaults to cache/faces")
    ap.add_argument("--img-size", type=int, default=224,
                    help="square frame size. 300 gives EfficientNet a 10x10 "
                         "feature map instead of 7x7, so concept regions get "
                         "3x4 cells for the mouth rather than 2x3.")
    args = ap.parse_args()

    paths = pd.read_csv(MASTER_CSV)["video_path"].tolist()
    if args.limit:
        paths = paths[: args.limit]
    items = [(p, args.cache_root, args.detect, args.img_size) for p in paths]

    fn, desc = (verify_one, "verify") if args.verify else (build_one, "cache")
    counts, errors = run(fn, items, args.workers, desc)

    print(f"\n{desc}: " + "  ".join(f"{k}={v}" for k, v in sorted(counts.items())))
    for path, status, detail in errors[:20]:
        print(f"  {status}: {path} {detail or ''}")
    if len(errors) > 20:
        print(f"  ... and {len(errors) - 20} more")

    if not args.verify:
        total = sum(
            cache_path_for(p, args.cache_root).stat().st_size
            for p in paths
            if cache_path_for(p, args.cache_root).exists()
        )
        print(f"cache size: {total / 2**30:.2f} GiB")


if __name__ == "__main__":
    main()
