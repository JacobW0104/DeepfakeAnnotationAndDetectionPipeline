import argparse

import numpy as np
import pandas as pd

RAW_CSV = "data/concept_labels_raw.csv"
MASTER_CSV = "data/faceforensics_concept_dataset.csv"
OUT_CSV = "data/faceforensics_concept_dataset_v3_snippets.csv"

CONCEPTS = [
    "lip_sync",
    "blinking",
    "facial_boundary",
    "texture",
    "identity",
    "lighting",
]
OUT_COLS = [f"c_{c}" for c in CONCEPTS]


def logit(p, eps=1e-4):
    """Log-odds of a probability, clipped away from 0 and 1."""
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))


def main():
    """Score the raw labels and write the training target CSV."""
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--transform",
        choices=["zscore", "rank", "raw"],
        default="rank",
        help="zscore: mean 0 sd 1 (NEGATIVE values -- needs a linear concept "
        "head with MSE, not the current sigmoid/BCE one). "
        "rank: per-concept rank mapped to [0,1], safe for sigmoid/BCE. "
        "raw: signed log-odds delta, unscaled.",
    )
    ap.add_argument("--agg", choices=["mean", "max"], default="mean",
                    help="how to combine the 3 snippets")
    ap.add_argument("--per-snippet", action="store_true",
                    help="one row per (video, snippet) instead of per video. "
                         "40-71%% of label variance is disagreement between "
                         "the 3 snippets of one video, so averaging them "
                         "discards two thirds of the supervision and blurs "
                         "artifacts that genuinely vary across a clip.")
    ap.add_argument("--out", default=OUT_CSV)
    args = ap.parse_args()

    long = pd.read_csv(RAW_CSV)
    print(f"{len(long)} rows, {long['video_path'].nunique()} videos")

    long["delta"] = logit(long["p_fake"]) - logit(long["p_real"])

    keys = ["video_path", "snippet"] if args.per_snippet else ["video_path"]
    wide = (
        long.groupby(keys + ["concept"])["delta"]
        .agg("mean" if args.per_snippet else args.agg)
        .unstack()
        .reindex(columns=CONCEPTS)
    )

    print("\nraw log-odds delta per concept:")
    print(wide.describe().T[["mean", "std", "min", "max"]].round(3).to_string())

    if args.transform == "zscore":
        scored = (wide - wide.mean()) / wide.std()
    elif args.transform == "rank":
        scored = wide.rank(pct=True)
    else:
        scored = wide

    scored.columns = OUT_COLS
    scored = scored.reset_index()

    master = pd.read_csv(MASTER_CSV)[
        ["video_path", "reference_path", "method", "is_fake"]
    ]
    if args.per_snippet:
        n_snip = int(long["snippet"].max()) + 1
        master = master.loc[master.index.repeat(n_snip)].copy()
        master["snippet"] = list(range(n_snip)) * (len(master) // n_snip)
        out = master.merge(scored, on=["video_path", "snippet"], how="left")
    else:
        out = master.merge(scored, on="video_path", how="left")
    fill = 0.5 if args.transform == "rank" else 0.0
    out[OUT_COLS] = out[OUT_COLS].fillna(fill)

    out.to_csv(args.out, index=False)
    print(f"\nwrote {args.out}  ({len(out)} rows, transform={args.transform}, "
          f"agg={args.agg})")
    print(out[OUT_COLS].describe().T[["mean", "std", "min", "max"]].round(4).to_string())

    corr = out.loc[out.is_fake == 1, OUT_COLS].corr()
    off = corr.values[~np.eye(len(OUT_COLS), dtype=bool)]
    print(f"\nmean off-diagonal concept correlation: {off.mean():.3f}")
    print("(old labels: 0.50 -- lower means the concepts became separable)")

    if args.transform == "zscore":
        print("\nWARNING: z-scored targets are negative for ~half the rows.")
        print("The current concept head is sigmoid/BCE and cannot represent")
        print("them. Switch to a linear head with MSE, or use --transform rank.")


if __name__ == "__main__":
    main()
