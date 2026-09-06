from pathlib import Path

import cv2
import numpy as np
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

CACHE_ROOT = Path("cache/faces")
YUNET_MODEL = "models/face_detection_yunet_2023mar.onnx"

_YUNET = None


def cache_path_for(video_path, root=None):
    """Cache file path for a video's decoded crops."""
    return Path(root or CACHE_ROOT) / Path(video_path).with_suffix(".npy")


def _yunet():
    """construct the per-process YuNet face detector."""
    global _YUNET
    if _YUNET is None:
        _YUNET = cv2.FaceDetectorYN.create(
            YUNET_MODEL, "", (320, 320), 0.6, 0.3, 5000
        )
    return _YUNET


def detect_face_box(frame, pad=0.30):
    """Padded face box for a frame, or None when no face is found."""
    h, w = frame.shape[:2]
    det = _yunet()
    det.setInputSize((w, h))
    _, faces = det.detect(frame)
    if faces is None or len(faces) == 0:
        return None
    best = faces[int(faces[:, -1].argmax())]
    x, y, bw, bh = best[:4]
    cx, cy = x + bw / 2.0, y + bh / 2.0
    side = max(bw, bh) * (1.0 + 2.0 * pad)
    x1, y1 = int(max(0, cx - side / 2)), int(max(0, cy - side / 2))
    x2, y2 = int(min(w, cx + side / 2)), int(min(h, cy + side / 2))
    if x2 - x1 < 32 or y2 - y1 < 32:
        return None
    return x1, y1, x2, y2


def center_face_crop(frame):
    """Fixed central crop, used when no face is detected."""
    h_img, w_img = frame.shape[:2]
    cw, ch = int(w_img * 0.55), int(h_img * 0.60)
    x1 = max(0, w_img // 2 - cw // 2)
    y1 = int(h_img * 0.05)
    return frame[y1 : min(h_img, y1 + ch), x1 : min(w_img, x1 + cw)]


def probe_video(video_path):
    """Return a video's frame count and frame rate."""
    cap = cv2.VideoCapture(str(video_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.release()
    return total_frames, fps


def snippet_frame_indices(
    total_frames,
    fps,
    num_snippets=3,
    frames_per_snippet=4,
    time_step=0.1,
):
    """Frame indices for each snippet of a video."""
    safe_duration = max(0.5, total_frames / fps - 0.5)
    fractions = [0.20, 0.50, 0.80][:num_snippets]
    targets = []
    for frac in fractions:
        start_t = safe_duration * frac
        for f_idx in range(frames_per_snippet):
            target_time = start_t + (f_idx * time_step)
            targets.append(min(int(target_time * fps), total_frames - 1))
    return targets


def decode_frames_uint8(video_path, targets, img_size=(224, 224), detect=False):
    """Decode the given frame indices of a video as face crops."""
    cap = cv2.VideoCapture(str(video_path))
    wanted = sorted(set(targets))
    cap.set(cv2.CAP_PROP_POS_FRAMES, wanted[0])
    pos = wanted[0]
    grabbed = {}
    for idx in wanted:
        while pos < idx:
            if not cap.grab():
                break
            pos += 1
        ret, frame = cap.read()
        pos += 1
        grabbed[idx] = frame if ret else None

    cap.release()

    out = np.empty((len(targets), img_size[1], img_size[0], 3), dtype=np.uint8)
    for i, target_frame in enumerate(targets):
        frame = grabbed[target_frame]
        if frame is None:
            out[i] = 0
        else:
            box = detect_face_box(frame) if detect else None
            if box is not None:
                x1, y1, x2, y2 = box
                frame = frame[y1:y2, x1:x2]
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if box is None:
                frame = center_face_crop(frame)
            out[i] = cv2.resize(frame, img_size)
    return out


def decode_clip_uint8(
    video_path,
    img_size=(224, 224),
    num_snippets=3,
    frames_per_snippet=4,
    time_step=0.1,
    detect=False,
):
    """Decode a whole clip's sampled frames as face crops."""
    total_frames, fps = probe_video(video_path)
    targets = snippet_frame_indices(
        total_frames, fps, num_snippets, frames_per_snippet, time_step
    )
    return decode_frames_uint8(video_path, targets, img_size, detect=detect)


class FaceForensicsAlignedDataset(Dataset):
    """Cached face crops and concept targets, one example per video or snippet."""

    def __init__(
        self,
        csv_file,
        num_frames=16,
        img_size=(224, 224),
        concept_cols=None,
        concept_scaler=None,
        cache_root=None,
        snippet_mode=False,
        frames_per_snippet=4,
        augment=False,
    ):
        """Load the label CSV and set up caching and concept scaling."""
        import pandas as pd

        self.df = pd.read_csv(csv_file)
        self.num_frames = num_frames
        self.img_size = img_size
        self.cache_root = cache_root
        self.snippet_mode = snippet_mode
        self.frames_per_snippet = frames_per_snippet
        self.augment = augment
        if snippet_mode and "snippet" not in self.df.columns:
            raise ValueError("snippet_mode needs a 'snippet' column "
                             "(build_concepts.py --per-snippet)")
        self.video_ids = pd.factorize(self.df["video_path"])[0]
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        self.concept_cols = concept_cols or [
            "c_lip_sync",
            "c_blinking",
            "c_facial_boundary",
            "c_texture",
            "c_identity",
            "c_lighting",
        ]

        if concept_scaler is None:
            p99 = self.df[self.concept_cols].quantile(0.99).values
            self.concept_scaler = np.maximum(p99, 1e-3).astype(np.float32)
        else:
            self.concept_scaler = np.asarray(concept_scaler, dtype=np.float32)

    def __len__(self):
        """Number of examples."""
        return len(self.df)

    def _extract_aligned_snippets(
            self, video_path, num_snippets=3, frames_per_snippet=4, time_step=0.1
    ):
        """Decode and cache a video's snippet frames."""
        return self._to_tensor(self._load_clip_uint8(video_path))

    def _load_clip_uint8(self, video_path):
        """Return a video's frames from cache, decoding on a miss."""
        cached = cache_path_for(video_path, self.cache_root)
        if cached.exists():
            return np.load(cached)
        return decode_clip_uint8(video_path, img_size=self.img_size)

    def _to_tensor(self, clip):
        """Convert a uint8 clip to a normalised tensor."""
        video = torch.from_numpy(clip).permute(0, 3, 1, 2).float() / 255.0
        return (video - self.mean) / self.std

    def _augment(self, clip):
        """Apply augmentation identically across a clip's frames."""
        r = np.random
        if r.rand() < 0.5:
            clip = clip[:, :, ::-1, :]
        if r.rand() < 0.3:
            q = [int(cv2.IMWRITE_JPEG_QUALITY), int(r.randint(40, 90))]
            clip = np.stack(
                [cv2.imdecode(cv2.imencode(".jpg", f, q)[1], 1) for f in clip]
            )
        if r.rand() < 0.2:
            k = int(r.choice([3, 5]))
            clip = np.stack([cv2.GaussianBlur(f, (k, k), 0) for f in clip])
        if r.rand() < 0.3:
            a, b = 1.0 + r.uniform(-0.2, 0.2), r.uniform(-20, 20)
            clip = np.clip(clip.astype(np.float32) * a + b, 0, 255).astype(np.uint8)
        return np.ascontiguousarray(clip)

    def __getitem__(self, idx):
        """Return one example's frames, label, concept targets and video id."""
        row = self.df.iloc[idx]
        clip = self._load_clip_uint8(row["video_path"])
        if self.snippet_mode:
            s = int(row["snippet"]) * self.frames_per_snippet
            clip = clip[s : s + self.frames_per_snippet]
        if self.augment:
            clip = self._augment(clip)
        video_tensor = self._to_tensor(clip)
        is_fake = torch.tensor(row["is_fake"], dtype=torch.float32)
        raw = row[self.concept_cols].values.astype(np.float32)
        concepts = torch.from_numpy(
            np.clip(raw / self.concept_scaler, 0.0, 1.0)
        )
        video_id = torch.tensor(self.video_ids[idx], dtype=torch.long)
        return video_tensor, is_fake, concepts, video_id


class SnippetPositionalEncoding(nn.Module):
    """Adds snippet and frame position embeddings to frame tokens."""

    def __init__(
        self,
        d_model: int,
        num_snippets: int = 3,
        frames_per_snippet: int = 4,
    ):
        """Create the snippet and frame embedding tables."""
        super().__init__()
        self.num_snippets = num_snippets
        self.frames_per_snippet = frames_per_snippet

        self.snippet_embed = nn.Embedding(num_snippets, d_model)
        self.frame_embed = nn.Embedding(frames_per_snippet, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional embeddings to a batch of frame tokens."""
        B, T, D = x.shape

        snippet_ids = torch.arange(
            self.num_snippets, device=x.device
        ).repeat_interleave(self.frames_per_snippet)

        frame_ids = torch.arange(
            self.frames_per_snippet, device=x.device
        ).repeat(self.num_snippets)

        pos_info = (
            self.snippet_embed(snippet_ids) + self.frame_embed(frame_ids)
        ).unsqueeze(0)

        return x + pos_info



CONCEPT_REGIONS = {
    "lip_sync": (0.34, 0.58, 0.68, 0.76),
    "blinking": (0.30, 0.34, 0.70, 0.51),
    "facial_boundary": (0.00, 0.00, 1.00, 1.00),
    "texture": (0.20, 0.46, 0.80, 0.60),
    "identity": (0.22, 0.28, 0.78, 0.82),
    "lighting": (0.00, 0.00, 1.00, 1.00),
}
CONCEPT_ORDER = list(CONCEPT_REGIONS)


def region_to_grid(region, size):
    """Map a relative region box to feature-map cell indices."""
    x0, y0, x1, y1 = region
    c0 = int(round(x0 * size))
    c1 = max(c0 + 1, int(round(x1 * size)))
    r0 = int(round(y0 * size))
    r1 = max(r0 + 1, int(round(y1 * size)))
    return r0, min(r1, size), c0, min(c1, size)


class EfficientNetTransformerDetector(nn.Module):
    """Predicts an authenticity logit and six concept logits from one clip."""

    def __init__(
        self,
        model_name: str = "efficientnet_b0",
        d_model: int = 512,
        nhead: int = 8,
        num_layers: int = 4,
        num_concepts: int = 6,
        dropout: float = 0.3,
        num_snippets: int = 3,
        frames_per_snippet: int = 4,
        concept_head_mode: str = "split",
        concept_features: str = "region",
    ):
        """Build the backbone, encoder, classification head and concept heads."""
        super().__init__()

        self.backbone = timm.create_model(
            model_name, pretrained=True, num_classes=0
        )
        in_features = self.backbone.num_features
        self.proj = nn.Linear(in_features, d_model)

        self.pos_encoder = SnippetPositionalEncoding(
            d_model=d_model,
            num_snippets=num_snippets,
            frames_per_snippet=frames_per_snippet,
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )

        self.dropout = nn.Dropout(dropout)
        self.cls_head = nn.Linear(d_model, 1)
        assert concept_head_mode in ("split", "shared")
        self.concept_head_mode = concept_head_mode
        assert concept_features in ("region", "region_motion", "pooled")
        self.concept_features = concept_features
        self.num_snippets = num_snippets
        self.frames_per_snippet = frames_per_snippet
        if concept_head_mode == "shared":
            self.concept_head = nn.Sequential(
                nn.Linear(d_model, 64),
                nn.ReLU(),
                nn.Linear(64, num_concepts),
            )
        else:
            if concept_features == "region":
                head_in = in_features
            elif concept_features == "region_motion":
                head_in = in_features * 2
            else:
                head_in = d_model
            self.concept_heads = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Linear(head_in, 32),
                        nn.ReLU(),
                        nn.Linear(32, 1),
                    )
                    for _ in range(num_concepts)
                ]
            )

    def concept_parameters(self):
        """Concept-head parameters, whichever head mode is active."""
        if self.concept_head_mode == "shared":
            return self.concept_head.parameters()
        return self.concept_heads.parameters()

    def freeze_backbone(self):
        """Disable gradients for the whole backbone."""
        for param in self.backbone.parameters():
            param.requires_grad = False

    def unfreeze_top_blocks(self, num_blocks: int = 2):
        """Re-enable gradients for the top backbone blocks only."""
        self.freeze_backbone()
        total_blocks = len(self.backbone.blocks)
        for block in self.backbone.blocks[total_blocks - num_blocks :]:
            for param in block.parameters():
                param.requires_grad = True

    def forward(self, x: torch.Tensor):
        """Return the authenticity logit and the six concept logits."""
        if x.dim() == 6:
            B, N_snip, N_frame, C, H, W = x.shape
            T = N_snip * N_frame
            x_spatial = x.view(B * T, C, H, W)
        else:
            B, T, C, H, W = x.shape
            x_spatial = x.view(B * T, C, H, W)

        feat_map = self.backbone.forward_features(x_spatial)
        features = feat_map.mean(dim=(2, 3))
        features = self.proj(features).view(B, T, -1)

        features = self.pos_encoder(features)

        trans_out = self.transformer(features)

        pooled = self.dropout(trans_out.mean(dim=1))
        logits = self.cls_head(pooled)

        if self.concept_features in ("region", "region_motion"):
            size = feat_map.shape[-1]
            if self.concept_features == "region_motion":
                ns, nf = self.num_snippets, self.frames_per_snippet
                assert T == ns * nf, (
                    f"region_motion needs T ({T}) == num_snippets ({ns}) * "
                    f"frames_per_snippet ({nf})"
                )
            outs = []
            for i, name in enumerate(CONCEPT_ORDER[: len(self.concept_heads)]):
                r0, r1, c0, c1 = region_to_grid(CONCEPT_REGIONS[name], size)
                f = feat_map[:, :, r0:r1, c0:c1].mean(dim=(2, 3))
                f = f.view(B, T, -1)
                if self.concept_features == "region_motion":
                    g = f.view(B, ns, nf, -1)
                    motion = (g[:, :, 1:] - g[:, :, :-1]).abs().mean(dim=(1, 2))
                    mean = f.mean(dim=1)
                    scale = mean.abs().mean(dim=-1, keepdim=True) + 1e-5
                    f = torch.cat([mean, motion / scale], dim=-1)
                else:
                    f = f.mean(dim=1)
                outs.append(self.concept_heads[i](self.dropout(f)))
            concepts = torch.cat(outs, dim=-1)
        elif self.concept_head_mode == "shared":
            concepts = self.concept_head(pooled)
        else:
            concepts = torch.cat(
                [head(pooled) for head in self.concept_heads], dim=-1
            )

        return logits, concepts


def _batch_corr(x, eps=1e-5):
    """Pearson correlation across the batch dimension."""
    x = x - x.mean(dim=0, keepdim=True)
    x = x / (x.std(dim=0, keepdim=True) + eps)
    return (x.T @ x) / x.shape[0]


class MaskedMultiTaskLoss(nn.Module):
    """Classification, concept and decorrelation loss terms combined."""

    def __init__(
        self,
        lambda_concept=0.5,
        pos_weight=0.2,
        concept_loss="kl",
        lambda_decorr=0.3,
    ):
        """Set the loss weights and the concept objective."""
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight))
        self.mse = nn.MSELoss(reduction="none")
        self.lambda_concept = lambda_concept
        assert concept_loss in ("mse", "bce", "kl")
        self.concept_loss = concept_loss
        self.lambda_decorr = lambda_decorr

    def forward(
        self, pred_logits, pred_concepts, target_is_fake, target_concepts
    ):
        """Return the total loss and its three components."""
        loss_cls = self.bce(pred_logits.squeeze(-1), target_is_fake)

        if self.concept_loss in ("bce", "kl"):
            raw = F.binary_cross_entropy_with_logits(
                pred_concepts, target_concepts, reduction="none"
            )
            if self.concept_loss == "kl":
                t = target_concepts.clamp(1e-6, 1 - 1e-6)
                entropy = -(t * t.log() + (1 - t) * (1 - t).log())
                raw = (raw - entropy).clamp_min(0.0)
        else:
            raw = self.mse(torch.sigmoid(pred_concepts), target_concepts)
        is_real = target_is_fake == 0
        has_concepts = target_concepts.sum(dim=-1) > 0
        mask = (is_real | has_concepts).float().unsqueeze(-1)

        masked = raw * mask
        loss_concept = masked.sum() / (
            mask.sum() * pred_concepts.size(-1) + 1e-8
        )

        loss_decorr = pred_logits.new_zeros(())
        if self.lambda_decorr and pred_concepts.shape[0] > 2:
            c_pred = _batch_corr(pred_concepts)
            c_targ = _batch_corr(target_concepts)
            off = ~torch.eye(
                pred_concepts.shape[-1],
                dtype=torch.bool,
                device=pred_concepts.device,
            )
            loss_decorr = ((c_pred - c_targ)[off] ** 2).mean()

        total_loss = (
            loss_cls
            + self.lambda_concept * loss_concept
            + self.lambda_decorr * loss_decorr
        )
        return total_loss, loss_cls, loss_concept, loss_decorr
