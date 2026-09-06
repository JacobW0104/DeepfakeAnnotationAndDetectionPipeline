"""Train the joint detection and concept model."""

import os

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader
from tqdm import tqdm

from CNN_Classes_Transfer import (
    EfficientNetTransformerDetector,
    FaceForensicsAlignedDataset,
    MaskedMultiTaskLoss,
)

def mean_by_group(values, index, n_groups):
    """Average rows sharing an index."""
    flat = values.dim() == 1
    v = values.unsqueeze(-1) if flat else values
    out = torch.zeros(n_groups, v.shape[1], dtype=v.dtype)
    cnt = torch.zeros(n_groups, 1, dtype=v.dtype)
    out.index_add_(0, index, v)
    cnt.index_add_(0, index, torch.ones(v.shape[0], 1, dtype=v.dtype))
    out = out / cnt.clamp_min(1)
    return out.squeeze(-1) if flat else out


TRAIN_CSV = os.environ.get("TRAIN_CSV", "data/train_concepts.csv")
VAL_CSV = os.environ.get("VAL_CSV", "data/val_concepts.csv")
CHECKPOINT_PATH = os.environ.get(
    "CHECKPOINT_PATH", "best_deepfake_effnet_trans.pth"
)
CACHE_ROOT = os.environ.get("CACHE_ROOT") or None
SNIPPET_MODE = os.environ.get("SNIPPET_MODE", "0") == "1"
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "48" if SNIPPET_MODE else "16"))
MODEL_NAME = os.environ.get("MODEL_NAME", "efficientnet_b0")
IMG_SIZE = int(os.environ.get("IMG_SIZE", "224"))
AUGMENT = os.environ.get("AUGMENT", "0") == "1"
NUM_EPOCHS = int(os.environ.get("NUM_EPOCHS", "15"))
STAGE1_EPOCHS = 3


def train_one_epoch(model, dataloader, optimizer, criterion, device, epoch):
    """Run one training epoch and return the mean loss."""
    model.train()
    running_loss = 0.0
    pbar = tqdm(
        dataloader, desc=f"Epoch {epoch:02d}/{NUM_EPOCHS} [Train]", leave=True
    )

    for videos, is_fake, concepts, _ in pbar:
        videos, is_fake, concepts = (
            videos.to(device),
            is_fake.to(device),
            concepts.to(device),
        )

        optimizer.zero_grad()
        logits, pred_concepts = model(videos)
        loss, loss_cls, loss_concept, loss_decorr = criterion(
            logits, pred_concepts, is_fake, concepts
        )

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        running_loss += loss.item()
        pbar.set_postfix(
            {
                "loss": f"{loss.item():.4f}",
                "cls": f"{loss_cls.item():.4f}",
                "concept": f"{loss_concept.item():.4f}",
                "decorr": f"{loss_decorr.item():.4f}",
            }
        )

    return running_loss / len(dataloader)


def validate_one_epoch(model, dataloader, criterion, device, epoch):
    """Evaluate on validation and return loss, AUC, accuracy and concept r."""
    model.eval()
    running_loss = 0.0
    all_logits, all_labels = [], []
    all_pred_c, all_targ_c, all_vids = [], [], []
    pbar = tqdm(
        dataloader, desc=f"Epoch {epoch:02d}/{NUM_EPOCHS} [Val]  ", leave=True
    )

    with torch.no_grad():
        for videos, is_fake, concepts, vids in pbar:
            all_vids.append(vids)
            videos, is_fake, concepts = (
                videos.to(device),
                is_fake.to(device),
                concepts.to(device),
            )
            logits, pred_concepts = model(videos)
            loss, _, _, _ = criterion(logits, pred_concepts, is_fake, concepts)
            running_loss += loss.item()
            all_logits.append(logits.squeeze(-1).float().cpu())
            all_labels.append(is_fake.cpu())
            all_pred_c.append(torch.sigmoid(pred_concepts).float().cpu())
            all_targ_c.append(concepts.cpu())
            pbar.set_postfix({"val_loss": f"{loss.item():.4f}"})

    logits = torch.cat(all_logits)
    labels = torch.cat(all_labels)
    p = torch.cat(all_pred_c)
    t = torch.cat(all_targ_c)

    vids = torch.cat(all_vids)
    uniq, inv = torch.unique(vids, return_inverse=True)
    if len(uniq) < len(vids):
        n = len(uniq)
        logits = mean_by_group(logits, inv, n)
        labels = mean_by_group(labels, inv, n)
        p = mean_by_group(p, inv, n)
        t = mean_by_group(t, inv, n)

    auc = roc_auc_score(labels.numpy(), logits.numpy())
    acc = ((logits > 0).float() == labels).float().mean().item()

    p = p.numpy()
    t = t.numpy()
    keep = (labels.numpy() == 0) | (t.sum(axis=1) > 0)
    rs = []
    for i in range(t.shape[1]):
        pi, ti = p[keep, i], t[keep, i]
        rs.append(
            np.corrcoef(pi, ti)[0, 1]
            if pi.std() > 1e-9 and ti.std() > 1e-9
            else np.nan
        )
    return running_loss / len(dataloader), auc, acc, np.array(rs)


def main():
    """Train the model and save the best checkpoints."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_dataset = FaceForensicsAlignedDataset(
        TRAIN_CSV, img_size=(IMG_SIZE, IMG_SIZE), cache_root=CACHE_ROOT,
        snippet_mode=SNIPPET_MODE, augment=AUGMENT,
    )
    val_dataset = FaceForensicsAlignedDataset(
        VAL_CSV,
        img_size=(IMG_SIZE, IMG_SIZE),
        concept_scaler=train_dataset.concept_scaler,
        cache_root=CACHE_ROOT,
        snippet_mode=SNIPPET_MODE,
    )
    print(f"{MODEL_NAME} @ {IMG_SIZE}px  augment={AUGMENT}")
    print(f"snippet_mode={SNIPPET_MODE}  batch={BATCH_SIZE}  "
          f"train rows={len(train_dataset)}  val rows={len(val_dataset)}")
    print(f"cache root: {CACHE_ROOT or 'cache/faces'} -> {CHECKPOINT_PATH}")
    print("concept p99 divisors:", train_dataset.concept_scaler.round(4))

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=8,
        pin_memory=True,
        persistent_workers=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
        persistent_workers=True,
    )

    concept_features = os.environ.get("CONCEPT_FEATURES", "region")
    concept_head_mode = os.environ.get("CONCEPT_HEAD_MODE", "split")
    model = EfficientNetTransformerDetector(
        model_name=MODEL_NAME,
        d_model=512,
        num_snippets=1 if SNIPPET_MODE else 3,
        frames_per_snippet=4,
        concept_features=concept_features,
        concept_head_mode=concept_head_mode,
    ).to(device)
    print(f"concept_features: {concept_features}  "
          f"concept_head_mode: {concept_head_mode}")
    lambda_decorr = float(os.environ.get("LAMBDA_DECORR", "0.3"))
    criterion = MaskedMultiTaskLoss(lambda_decorr=lambda_decorr).to(device)
    print(f"lambda_decorr: {lambda_decorr}")
    best_val_auc = 0.0
    best_joint = -1.0
    bestauc_path = CHECKPOINT_PATH.replace(".pth", "_bestauc.pth")

    model.freeze_backbone()
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=2e-4,
        weight_decay=1e-2,
    )

    for epoch in range(1, NUM_EPOCHS + 1):
        if epoch == STAGE1_EPOCHS + 1:
            print("\n--> Transitioning to STAGE 2: Unfreezing top blocks...")
            model.unfreeze_top_blocks(num_blocks=2)
            optimizer = torch.optim.AdamW(
                [
                    {"params": model.backbone.parameters(), "lr": 1e-5},
                    {"params": model.proj.parameters(), "lr": 1e-4},
                    {"params": model.transformer.parameters(), "lr": 1e-4},
                    {"params": model.cls_head.parameters(), "lr": 1e-4},
                    {"params": model.concept_parameters(), "lr": 1e-4},
                ],
                weight_decay=1e-2,
            )

        train_loss = train_one_epoch(
            model, train_loader, optimizer, criterion, device, epoch
        )
        val_loss, val_auc, val_acc, val_r = validate_one_epoch(
            model, val_loader, criterion, device, epoch
        )
        concept_rs = " ".join(f"{r:.2f}" for r in val_r)
        print(
            f"Epoch {epoch:02d} | train {train_loss:.4f} | val {val_loss:.4f} "
            f"| AUC {val_auc:.4f} | acc {val_acc:.4f}"
            f" | concept r {np.nanmean(val_r):.3f} [{concept_rs}]"
        )

        payload = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "val_loss": val_loss,
            "val_auc": val_auc,
            "val_concept_r": val_r,
            "concept_scaler": train_dataset.concept_scaler,
            "model_name": MODEL_NAME,
            "img_size": IMG_SIZE,
            "concept_features": concept_features,
            "concept_head_mode": concept_head_mode,
        }

        joint = val_auc + np.nanmean(val_r)
        if joint > best_joint:
            best_joint = joint
            torch.save({**payload, "joint": joint}, CHECKPOINT_PATH)
            print(
                f"--> Saved best JOINT (AUC {val_auc:.4f} + r "
                f"{np.nanmean(val_r):.4f} = {joint:.4f})"
            )

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            torch.save(payload, bestauc_path)
            print(f"--> Saved best AUC ({val_auc:.4f})")


if __name__ == "__main__":
    main()
