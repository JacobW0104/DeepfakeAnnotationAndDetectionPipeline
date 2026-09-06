# Concept-Grounded Deepfake Detection

Code accompanying the MSc dissertation *Human-Centred Explainable Deepfake
Detection*. The system predicts a binary authenticity score for a video
together with six named concept scores (lip-sync distortion, blinking
anomalies, blending seams, skin-texture inconsistency, identity drift, lighting
mismatch), trained on concept labels generated automatically by a
vision-language model rather than by human annotators.

---

## 1. Requirements

Python 3.13 with a CUDA-capable GPU. The reported results were produced on a
single RTX 4070 (12 GB).

```
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Principal dependencies: PyTorch 2.11 (cu128), timm 1.0.28, transformers 5.15.1,
OpenCV, scikit-learn, pandas, numpy.

### Face detector weights

The YuNet face detector is not included and must be downloaded separately. Take
`face_detection_yunet_2023mar.onnx` from the OpenCV Zoo repository, under
`models/face_detection_yunet/`:

<https://github.com/opencv/opencv_zoo>

Place it at `models/face_detection_yunet_2023mar.onnx` relative to the project
root.

    file:    face_detection_yunet_2023mar.onnx
    size:    232,589 bytes
    sha256:  8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4

## 2. Data

FaceForensics++ must be obtained from the authors under their access terms. Place the corpus under `FaceForensics/` so that
original and manipulated videos are reachable by the paths recorded in the
master CSV.

The pipeline expects 1,000 original videos and 1,000 videos from each of
Deepfakes, FaceSwap, FaceShifter, Face2Face and NeuralTextures at the **c23**
compression level, giving 6,000 videos in total.

## 3. Pipeline

Each stage writes files consumed by the next. Stages 1 to 4 are run once;
stage 5 is repeated per experiment.

### Stage 1 — build the master video list and paired references

```
python comparative_labeler.py
```

Scans the corpus and writes `data/faceforensics_concept_dataset.csv`, which lists
every video, its unmanipulated reference, its manipulation method and its
authenticity label. Every later stage reads its video list from this file.

### Stage 2 — cache face crops

```
python preprocess_faces.py --detect --img-size 300 --cache-root cache/faces_det300 --workers 8
```

Samples three snippets per video at the 20%, 50% and 80% points of its
duration, four frames per snippet at 0.10 s spacing, detects the face with
YuNet, expands the box by 0.25 and writes 300 x 300 crops to the cache
(approximately 18 GB). Omit `--detect` to build a
centre-crop cache instead, which is the comparison reported in the ablations.

### Stage 3 — annotate concepts

```
python relabel.py --dump-regions      # inspect the six region crops first
python relabel.py --limit 200         # trial run, then audit
python relabel.py --audit
python relabel.py                     # full run
```

Runs Qwen2-VL-2B-Instruct over every manipulated video, asking one yes/no
question per concept about a pair of region crops taken from the manipulated
clip and its reference at matched frame indices. Writes the raw `p_real` and
`p_fake` probabilities per video, concept and snippet to
`data/concept_labels_raw.csv`. No score is derived at this stage.

### Stage 4 — score the labels and split the corpus

```
python build_concepts.py --per-snippet
set SPLIT=three
set MASTER_CSV=data/faceforensics_concept_dataset_v3_snippets.csv
python main.py
```

`build_concepts.py` converts the stored probabilities into training targets: a
signed difference in log-odds, then a per-concept percentile rank in [0, 1].
`main.py` groups videos by the identities their filenames encode, then assigns
whole groups to subsets, writing `data/train_concepts.csv`, `val_concepts.csv` and `test_concepts.csv` (400 / 50 / 50 groups, 14,400 / 1,800 / 1,800 snippet
rows).

> `main.py` performs the split at import time and overwrites those three CSVs.
> Do not import it from other code.

### Stage 5 — train

```
set MODEL_NAME=efficientnet_b3
set IMG_SIZE=300
set AUGMENT=1
set SNIPPET_MODE=1
set BATCH_SIZE=24
set CACHE_ROOT=cache/faces_det300
set LAMBDA_DECORR=0.1
set CHECKPOINT_PATH=detector.pth
python trainer.py
```

Fifteen epochs. Two checkpoints are written: the best joint
score (validation AUC plus mean concept correlation), which is the reported
model, and the best AUC alone, suffixed `_bestauc`.

### Stage 6 — evaluate

```
set VAL_CSV=data/val_concepts.csv
python evaluate.py --checkpoint detector.pth

set VAL_CSV=data/test_concepts.csv
python evaluate.py --checkpoint detector.pth --threshold 1.9071 --dump test_predictions.npz
```

Reports AUC with a bootstrap confidence interval over identity groups, average
precision, operating points, per-concept correlations, the predicted and target
inter-concept correlation matrices, and strongest-artefact accuracy. `--dump`
writes per-video predictions so metrics and intervals can be recomputed without
re-running inference.

The threshold is fitted on validation by maximising Youden's J and transferred
unchanged to the test set; fitting it on test would bias the operating point.

## 4. Environment variables

| Variable | Default | Effect |
|---|---|---|
| `MODEL_NAME` | `efficientnet_b0` | timm backbone |
| `IMG_SIZE` | `224` | input resolution |
| `AUGMENT` | `0` | flip, JPEG recompression, blur, brightness jitter |
| `SNIPPET_MODE` | `0` | one snippet per example rather than the whole video |
| `BATCH_SIZE` | 48 snippet / 16 video | |
| `CACHE_ROOT` | `cache/faces` | which crop cache to read |
| `LAMBDA_DECORR` | `0.3` | weight of the decorrelation term |
| `CONCEPT_FEATURES` | `region` | `region`, `pooled`, or `region_motion` |
| `CONCEPT_HEAD_MODE` | `split` | `split` or `shared` |
| `TRAIN_CSV`, `VAL_CSV` | `data/train_concepts.csv`, `data/val_concepts.csv` | |
| `CHECKPOINT_PATH` | `best_deepfake_effnet_trans.pth` | where trainer.py writes its weights |
| `NUM_EPOCHS` | `15` | |
| `SPLIT` | `two` | `three` adds a held-out test subset (`main.py`) |
| `MASTER_CSV` | `data/faceforensics_concept_dataset.csv` | video list (`main.py`) |

## 5. Reproducing the reported experiments

All ablations use the stage 5 command with one variable changed, a distinct
`CHECKPOINT_PATH`, and are evaluated on validation.

| Experiment | Change |
|---|---|
| Final configuration | as stage 5 |
| Centre crop | `CACHE_ROOT=cache/faces_centre300` (build that cache without `--detect`) |
| Reduced capacity | `MODEL_NAME=efficientnet_b0 IMG_SIZE=224 CACHE_ROOT=cache/faces_det` |
| Whole-video examples | `SNIPPET_MODE=0 BATCH_SIZE=8` with video-level CSVs |
| No augmentation | `AUGMENT=0` |
| Pooled concept features | `CONCEPT_FEATURES=pooled` |
| Shared concept heads | `CONCEPT_HEAD_MODE=shared CONCEPT_FEATURES=pooled` |
| Motion features | `CONCEPT_FEATURES=region_motion` |
| No decorrelation | `LAMBDA_DECORR=0` |

Shared heads are only compatible with pooled features: a shared head produces
six outputs from one vector and cannot read six distinct regions.

Video-level CSVs for the whole-video comparison are produced by running
`build_concepts.py` without `--per-snippet`, then `main.py` against that file,
renaming its outputs before they overwrite the snippet-level splits.

Replicate runs of an identical configuration differ by up to 0.06 in an
individual concept correlation, 0.009 in AUC and 2.2 percentage points in
artefact identification. Differences smaller than these are not interpretable.

The repository ships with the annotation already performed, so stages 1 and 3
can be skipped. `data/concept_labels_raw.csv` holds the 90,000 raw annotator
probabilities and `data/faceforensics_concept_dataset.csv` the master video
list. Stage 4 must be run to derive the scored labels and the splits from them;
it takes seconds and is deterministic under the fixed seed, so it reproduces
the exact partition used for the reported results. Stage 2, the crop cache, must also be rebuilt locally, since the video data is
not redistributed, and stage 5 must be run to produce the model weights,
which are likewise not distributed.

## 6. File map

| File | Role |
|---|---|
| `comparative_labeler.py` | builds the master video list and reference pairing |
| `frame_grabber.py` | frame sampling helpers used by the labeller |
| `preprocess_faces.py` | face detection and crop caching |
| `relabel.py` | concept annotation with Qwen2-VL; region definitions |
| `build_concepts.py` | log-odds differencing and rank scoring |
| `main.py` | identity grouping and train/validation/test split |
| `identity_groups.py` | read-only reimplementation of the grouping, for evaluation |
| `CNN_Classes_Transfer.py` | dataset, model, region definitions, loss |
| `trainer.py` | training loop, two-stage schedule, checkpoint selection |
| `evaluate.py` | metrics, bootstrap intervals, prediction dumps |
