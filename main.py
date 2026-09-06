import os
from pathlib import Path

import numpy as np
import pandas as pd

parent = {}


def find(x):
    """Union-find root of an identity."""
    parent.setdefault(x, x)
    while parent[x] != x:
        parent[x] = parent[parent[x]]
        x = parent[x]
    return x


def union(a, b):
    """Merge two identities into one group."""
    ra, rb = find(a), find(b)
    if ra != rb:
        parent[ra] = rb


def identities(path):
    """Identity tokens encoded in a video filename."""
    return Path(path).stem.split("_")


MASTER_CSV = os.environ.get(
    "MASTER_CSV", "data/faceforensics_concept_dataset.csv"
)
df = pd.read_csv(MASTER_CSV)
print(f"source: {MASTER_CSV}")

for path in df["video_path"]:
    parts = identities(path)
    for other in parts[1:]:
        union(parts[0], other)

groups = df["video_path"].map(lambda p: find(identities(p)[0]))

unique = np.array(sorted(groups.unique()))
np.random.default_rng(42).shuffle(unique)

three_way = os.environ.get("SPLIT", "two") == "three"
n_hold = int(len(unique) * 0.20)
if three_way:
    test_groups = set(unique[: n_hold // 2])
    val_groups = set(unique[n_hold // 2 : n_hold])
else:
    test_groups = set()
    val_groups = set(unique[:n_hold])

is_test = groups.isin(test_groups)
is_val = groups.isin(val_groups)
train_df = df[~(is_val | is_test)]
val_df = df[is_val]
test_df = df[is_test]

train_df.to_csv("data/train_concepts.csv", index=False)
val_df.to_csv("data/val_concepts.csv", index=False)
if three_way:
    test_df.to_csv("data/test_concepts.csv", index=False)

print(f"Train samples: {len(train_df)} | Val samples: {len(val_df)}")
print(
    f"Train fake ratio: {train_df['is_fake'].mean():.4f} | "
    f"Val fake ratio: {val_df['is_fake'].mean():.4f}"
)
print(
    f"Identity groups: {len(unique)} "
    f"({len(unique) - len(val_groups) - len(test_groups)} train / "
    f"{len(val_groups)} val / {len(test_groups)} test)"
)
if three_way:
    print(f"Test samples: {len(test_df)} | "
          f"Test fake ratio: {test_df['is_fake'].mean():.4f}")
