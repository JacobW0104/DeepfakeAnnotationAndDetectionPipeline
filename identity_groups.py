from pathlib import Path

import pandas as pd

MASTER_CSV = "data/faceforensics_concept_dataset.csv"


def identities(path):
    """Identity tokens encoded in a video filename."""
    return Path(path).stem.split("_")


def _build(paths):
    """Union identities that co-occur and return a lookup function."""
    parent = {}

    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for path in paths:
        parts = identities(path)
        for other in parts[1:]:
            union(parts[0], other)
    return find


def group_labels(video_paths, master_csv=MASTER_CSV):
    """Group id for each given video path."""
    master = pd.read_csv(master_csv)["video_path"]
    find = _build(master)
    return [find(identities(p)[0]) for p in video_paths]
