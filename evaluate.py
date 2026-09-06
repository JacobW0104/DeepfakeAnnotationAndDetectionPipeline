import argparse
import os

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)
from torch.utils.data import DataLoader
from tqdm import tqdm

from CNN_Classes_Transfer import (
    EfficientNetTransformerDetector,
    FaceForensicsAlignedDataset,
)
from identity_groups import group_labels
from trainer import mean_by_group

VAL_CSV = os.environ.get("VAL_CSV", "data/val_concepts.csv")
SNIPPET_MODE = os.environ.get("SNIPPET_MODE", "0") == "1"
CONCEPT_COLS = [
    "c_lip_sync",
    "c_blinking",
    "c_facial_boundary",
    "c_texture",
    "c_identity",
    "c_lighting",
]


def collect(model, loader, device):
    """Run the model over a split and fold snippets into video-level arrays."""
    logits, labels, preds, targets, vids = [], [], [], [], []
    with torch.no_grad():
        for videos, is_fake, concepts, vid in tqdm(loader, desc="val"):
            vids.append(vid)
            out_logits, out_concepts = model(videos.to(device))
            logits.append(out_logits.squeeze(-1).float().cpu())
            preds.append(torch.sigmoid(out_concepts).float().cpu())
            labels.append(is_fake)
            targets.append(concepts)
    lo, la = torch.cat(logits), torch.cat(labels)
    pr, ta = torch.cat(preds), torch.cat(targets)

    v = torch.cat(vids)
    uniq, inv = torch.unique(v, return_inverse=True)
    if len(uniq) < len(v):
        n = len(uniq)
        print(f"  folding {len(v)} snippets -> {n} videos")
        lo, la = mean_by_group(lo, inv, n), mean_by_group(la, inv, n)
        pr, ta = mean_by_group(pr, inv, n), mean_by_group(ta, inv, n)
    return lo.numpy(), la.numpy(), pr.numpy(), ta.numpy()


def bootstrap_auc_ci(logits, labels, groups, n_boot=2000, alpha=0.05, seed=0):
    """Percentile bootstrap AUC interval, resampling identity groups."""
    rng = np.random.default_rng(seed)
    groups = np.asarray(groups)
    uniq = np.unique(groups)
    members = {g: np.flatnonzero(groups == g) for g in uniq}
    stats = []
    for _ in range(n_boot):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([members[g] for g in pick])
        y, s = labels[idx], logits[idx]
        if y.min() == y.max():
            continue
        stats.append(roc_auc_score(y, s))
    lo, hi = np.percentile(stats, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi), len(stats), len(uniq)


def report_threshold(name, logits, labels, thr):
    """Print the operating point at one decision threshold."""
    pred = (logits > thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(labels, pred, labels=[0, 1]).ravel()
    acc = (tp + tn) / len(labels)
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    tnr = tn / (tn + fp) if tn + fp else 0.0
    bal = (rec + tnr) / 2
    print(f"\n{name}  (logit > {thr:+.4f})")
    print(f"  accuracy {acc:.4f}   balanced {bal:.4f}   F1 {f1:.4f}")
    print(f"  precision {prec:.4f}  recall/TPR {rec:.4f}  specificity/TNR {tnr:.4f}")
    print(f"  confusion:  TN {tn:5d}  FP {fp:5d}")
    print(f"              FN {fn:5d}  TP {tp:5d}")


def main():
    """Load a checkpoint, evaluate it and print all metrics."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="best_deepfake_effnet_trans.pth")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--dump", default=None,
                    help="write per-video predictions, targets and labels to "
                         "this .npz so metrics can be re-derived, and "
                         "bootstrapped, without re-running inference.")
    ap.add_argument("--threshold", type=float, default=None,
                    help="Apply a threshold chosen elsewhere (i.e. on the "
                         "validation split) instead of fitting one here. A "
                         "threshold fitted on the data it is scored against "
                         "is optimistically biased.")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    print(f"checkpoint: {args.checkpoint}")
    print(f"  epoch {ckpt.get('epoch')}  val_loss {ckpt.get('val_loss'):.4f}"
          f"  val_auc {ckpt.get('val_auc'):.4f}")

    mode = ("shared" if "concept_head.0.weight" in ckpt["model_state_dict"]
            else "split")
    print(f"  concept head: {mode}")
    model_name = ckpt.get("model_name", "efficientnet_b0")
    img_size = int(ckpt.get("img_size", 224))
    print(f"  backbone: {model_name} @ {img_size}px")
    concept_features = ckpt.get("concept_features", "region")
    print(f"  concept features: {concept_features}")
    model = EfficientNetTransformerDetector(
        model_name=model_name, d_model=512,
        num_snippets=1 if SNIPPET_MODE else 3, frames_per_snippet=4,
        concept_head_mode=mode,
        concept_features=concept_features,
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    scaler = ckpt.get("concept_scaler")
    if scaler is None:
        scaler = np.ones(len(CONCEPT_COLS), dtype=np.float32)
        print("  (legacy checkpoint: no concept_scaler, using raw targets)")
    else:
        print("  concept_scaler:", np.asarray(scaler).round(4))

    ds = FaceForensicsAlignedDataset(
        VAL_CSV,
        img_size=(img_size, img_size),
        concept_scaler=scaler,
        cache_root=os.environ.get("CACHE_ROOT") or None,
        snippet_mode=SNIPPET_MODE,
    )
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.workers, pin_memory=True)
    logits, labels, preds, targets = collect(model, loader, device)

    if args.dump:
        np.savez(args.dump, logits=logits, labels=labels, preds=preds,
                 targets=targets,
                 video_paths=np.array(pd.factorize(ds.df["video_path"])[1]))
        print(f"  dumped per-video arrays to {args.dump}")

    n_fake, n_real = int(labels.sum()), int((labels == 0).sum())
    print(f"\nval set: {len(labels)} videos  {n_fake} fake / {n_real} real")

    auc = roc_auc_score(labels, logits)
    ap_score = average_precision_score(labels, logits)
    print(f"\nAUC {auc:.4f}   average precision {ap_score:.4f}"
          f"   (AP baseline = base rate {labels.mean():.4f})")

    order = pd.factorize(ds.df["video_path"])[1]
    groups = group_labels(order)
    if len(groups) == len(labels):
        lo_ci, hi_ci, n_ok, n_grp = bootstrap_auc_ci(logits, labels, groups)
        print(f"  95% CI [{lo_ci:.4f}, {hi_ci:.4f}]  (percentile bootstrap, "
              f"{n_ok} resamples over {n_grp} identity groups, "
              f"+/-{(hi_ci - lo_ci) / 2:.4f})")
    else:
        print(f"  (CI skipped: {len(groups)} paths vs {len(labels)} rows)")

    fpr, tpr, thresholds = roc_curve(labels, logits)
    youden = thresholds[np.argmax(tpr - fpr)]

    candidates = np.unique(logits)
    accs = [((logits > t).astype(int) == labels).mean() for t in candidates]
    best_acc_thr = candidates[int(np.argmax(accs))]

    if args.threshold is not None:
        report_threshold(
            f"TRANSFERRED threshold (fitted on validation)",
            logits, labels, args.threshold,
        )
    report_threshold("default threshold", logits, labels, 0.0)
    report_threshold("max balanced accuracy (Youden's J)", logits, labels, youden)
    report_threshold("max raw accuracy", logits, labels, best_acc_thr)

    print("\nNote: sigmoid(logit) is NOT calibrated P(fake) — pos_weight=0.2")
    print("shifts the optimum. Use the threshold, not the probability.")

    print("\nconcept head (masked as in training: reals + fakes with labels)")
    mask = (labels == 0) | (targets.sum(axis=1) > 0)
    p, t = preds[mask], targets[mask]
    print(f"  {mask.sum()} of {len(labels)} rows in mask")
    print(f"  {'concept':<20} {'pred_mean':>10} {'targ_mean':>10} "
          f"{'MAE':>8} {'pred_std':>9} {'pearson_r':>10}")
    for i, col in enumerate(CONCEPT_COLS):
        pi, ti = p[:, i], t[:, i]
        r = np.corrcoef(pi, ti)[0, 1] if pi.std() > 1e-9 and ti.std() > 1e-9 else float("nan")
        print(f"  {col:<20} {pi.mean():10.4f} {ti.mean():10.4f} "
              f"{np.abs(pi - ti).mean():8.4f} {pi.std():9.5f} {r:10.4f}")
    print("\n  pred_std near 0 means the head outputs a constant and has")
    print("  learned nothing; pearson_r is the metric that actually matters.")

    def corr_block(name, m):
        c = np.corrcoef(m, rowvar=False)
        off = c[~np.eye(len(CONCEPT_COLS), dtype=bool)]
        print(f"\n  {name} inter-concept correlation "
              f"(mean off-diagonal {off.mean():.3f}):")
        print("  " + " " * 18 + "".join(f"{col[2:11]:>10}" for col in CONCEPT_COLS))
        for i, col in enumerate(CONCEPT_COLS):
            row = "".join(f"{c[i, j]:>10.2f}" for j in range(len(CONCEPT_COLS)))
            print(f"  {col:<18}{row}")

    corr_block("TARGET", t)
    corr_block("PREDICTED", p)

    fake = labels == 1
    pf, tf = preds[fake], targets[fake]
    keep = tf.std(axis=1) > 1e-9
    top_hit = (pf[keep].argmax(axis=1) == tf[keep].argmax(axis=1)).mean()
    print("")
    print(f"  strongest artefact identified: {top_hit:.4f} "
          f"on {int(keep.sum())} manipulated videos "
          f"(chance {1 / len(CONCEPT_COLS):.4f})")
    print("\n  Predicted correlation much HIGHER than target means the head is")
    print("  collapsing the concepts back into a single shared factor.")


if __name__ == "__main__":
    main()
