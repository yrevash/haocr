# Human & Animal Detection + Offline OCR Pipeline

> AI Technical Assignment — Computer Vision & OCR (Offline Deployment)

**Quick links:**
- [WandB Training Run](https://api.wandb.ai/links/yrevash-student/063chsq1) — full training logs, loss curves, accuracy charts
- [Streamlit App](https://haocri.streamlit.app/) — interactive UI for both detection and OCR
- [Dataset on Roboflow](https://universe.roboflow.com/yrevash-7tnv7/humans-and-animals-detection-vv9pj) — 17,756 images, custom annotations

---

## What This Project Does

Two separate tasks, both running fully offline with no cloud APIs:

**Part A** — Detects humans and animals in video using a two-model pipeline (detector + classifier). Drop videos into `test_videos/`, get annotated outputs in `outputs/`.

**Part B** — Reads stenciled/painted text from industrial and military-style boxes using offline OCR. Handles faded paint, low contrast, surface damage, and rotated images.

---

## How to Run

### Setup

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

PaddleOCR downloads its models on first run (~200MB), then works fully offline after that.

### Detection (CLI)

```bash
# Processes all videos in test_videos/ → saves annotated videos to outputs/
python main.py
```

There's a `PROCESS_SECONDS` variable at the top of `main.py` — set it to `3`, `5`, `10` for quick tests, or `None` to process the full video.

### OCR (CLI)

```bash
python ocr/pipeline.py                      # test_images/ → outputs/
python ocr/pipeline.py some_folder outputs  # custom input/output paths
```

### Streamlit App

```bash
python -m streamlit run app.py
```

(Use `python -m` to make sure it uses the venv's Python, not the system one.)

The app has three tabs:
1. **Human & Animal Detection** — pick a sample video or upload your own, choose duration, run detection, watch the result in-browser
2. **Industrial OCR** — pick a sample image or upload your own, run OCR, see annotated regions + extracted text + JSON export
3. **Results Browser** — browse previously generated outputs without re-running anything

---

## Part A: Human & Animal Detection

### The Approach

The assignment required **two separate models** (no YOLO), so I went with:

1. **Faster R-CNN (ResNet50-FPN)** — the detector. Finds bounding boxes for humans and animals in each frame. Two-stage detector, good localization accuracy, handles objects at different scales thanks to the FPN neck.

2. **EfficientNet-B3** — the classifier. Each crop from the detector gets classified as either Human or Animal. Lightweight enough for real-time crop classification, and compound scaling means it punches above its weight for the parameter count.

The pipeline runs in two stages per frame:
- Stage 1: Faster R-CNN detects all objects, NMS removes duplicate boxes
- Stage 2: Each detection is cropped and passed through EfficientNet for classification
- Combined confidence = detection_score * classifier_score

### Dataset

Used a custom dataset from Roboflow — **"Humans and Animals Detection"** with 17,756 images. Not COCO, not ImageNet.

- **Train**: 15,754 images (89%)
- **Validation**: 1,013 images (6%)
- **Test**: 989 images (6%)

The raw dataset was heavily imbalanced (794 humans vs 14,874 animals), so the training script balances it by capping animal samples at 2,500 and keeping all human samples. After balancing: 3,294 training samples, 532 validation samples.

**Why this dataset?** It has bounding box annotations for both humans and animals in real-world scenes. The images are diverse (indoor, outdoor, different lighting), and it's hosted on Roboflow so the download/format conversion is straightforward.

### Training

Training was done on GPU (CUDA) with full WandB logging. The training script is at `datasets/train.py`.

**Detector (Faster R-CNN):**
- 10 epochs, SGD optimizer (lr=0.005, momentum=0.9)
- StepLR scheduler (step=3, gamma=0.1)
- Started from COCO-pretrained backbone weights, replaced the head for 3 classes (background, human, animal)
- Final loss: **0.1012**

**Classifier (EfficientNet-B3):**
- 15 epochs, AdamW optimizer (lr=0.0003, weight_decay=0.01)
- CosineAnnealing LR schedule
- Weighted cross-entropy to handle class imbalance (human weight=2.56, animal weight=0.62)
- Data augmentation: random crop, horizontal flip, rotation, color jitter
- Best validation accuracy: **98.13%**

All metrics (loss, accuracy, learning rate, batch-level breakdowns) are logged to WandB:
[View the full training run here](https://wandb.ai/yrevash-student/human-animal-detection/runs/8v9x9i8e)

### Key Parameters

| Parameter | Value |
|-----------|-------|
| Detection confidence threshold | 0.6 |
| NMS IoU threshold | 0.3 |
| Classifier input size | 300x300 |
| Color coding | Green = HUMAN, Orange = ANIMAL |

---

## Part B: Offline OCR

### Engine

**PaddleOCR v3 (PP-OCRv5)** — fully offline after the initial model download. Has built-in text detection (DB++) and recognition (SVTR) in a single API. I tried Tesseract first but PaddleOCR does much better on degraded/stenciled text.

### Why It's Not Just "Run OCR"

Military box text is rough — faded paint, rust, low contrast, chipped stencil edges. A single OCR pass misses a lot. So the pipeline generates **4 enhanced versions** of each image and runs OCR on all of them:

1. **Original** — works for clean white stencils
2. **CLAHE grayscale** — pulls faded text from dark surfaces (clipLimit=3.0)
3. **Sharpened** — bilateral filter + unsharp mask recovers chipped stencil edges
4. **CLAHE color** — LAB space L-channel enhancement, keeps color features

Then it merges all results with three-stage deduplication:
1. **IoU + fuzzy text match** — same text in same region across variants? keep the best one
2. **Noise filter** — short low-confidence hits are usually rust/scratches, not text
3. **Fragment absorption** — if a word-level detection is inside a full-line detection, drop it

### Orientation Handling

Some test images are rotated (upside-down boxes, etc.). The pipeline uses a separate orientation detector to check the angle, then manually rotates the image and re-runs OCR. Keeps whichever result (rotated vs non-rotated) has higher average confidence. This prevents the orientation detector from being wrong and making things worse.

### Memory Management

High-res images (like `ocr_4.jpg` at 2560x1920) get capped at 1280px longest side before processing. PaddleOCR's feature maps grow quadratically with image size, and OCR accuracy doesn't improve much past ~1280px for stenciled text.

---

## Project Structure

```
maharshi/
├── main.py                  # CLI — processes test_videos/ → outputs/
├── app.py                   # Streamlit app (detection + OCR + results browser)
├── requirements.txt
├── datasets/
│   └── train.py             # Full training script with WandB logging
├── models/
│   ├── detector_final.pth   # Faster R-CNN (3 classes)
│   └── classifier_best.pth  # EfficientNet-B3 (2 classes)
├── ocr/
│   ├── __init__.py
│   └── pipeline.py          # OCR pipeline module
├── test_videos/             # Drop input videos here
│   ├── people.mp4
│   └── people_animal.mp4
├── test_images/             # OCR input images
│   ├── ocr_1.png            # Dark military box — faded text
│   ├── ocr_2.png            # Green mag box — white stencil
│   ├── ocr_3.png            # Ammo crate — multi-line text
│   └── ocr_4.jpg            # High-res ammo can (2560x1920)
└── outputs/                 # All generated outputs
    ├── *.mp4                # Annotated detection videos
    ├── *_ocr.jpg            # Annotated OCR images
    ├── *_ocr.json           # Per-image OCR results (structured)
    └── ocr_results.json     # Combined OCR results
```

---

## Training Script

The full training pipeline is in `datasets/train.py`. It:

1. Downloads the dataset from Roboflow
2. Balances the class distribution (caps animal samples at 2,500)
3. Trains Faster R-CNN for detection (10 epochs)
4. Extracts crops from ground-truth boxes to build a classification dataset
5. Trains EfficientNet-B3 for classification (15 epochs)
6. Logs everything to WandB throughout

To re-run training, you'd need a Roboflow API key and a GPU. The trained model weights (`detector_final.pth` and `classifier_best.pth`) are already included in `models/`.

---

## Challenges & What I'd Improve

**Class imbalance** — The dataset had way more animals than humans (~19:1). Handled it with sampling limits and weighted loss, but a bigger human dataset would help.

**Duplicate detections** — Faster R-CNN sometimes outputs overlapping boxes for the same object. Per-class NMS with IoU threshold 0.3 fixed this.

**Faded stencil text** — Single-pass OCR misses a lot on damaged surfaces. The multi-variant approach with CLAHE + sharpening + deduplication gets most of it, but severely faded text (like on `ocr_1.png`) still has occasional misreads.

**OCR orientation** — Enabling PaddleOCR's built-in orientation classifier actually fragments the text reads. Using it only for angle detection and doing the rotation manually works much better.

**Video playback in browser** — OpenCV writes mp4v codec which browsers can't play. Had to re-encode to H.264 via ffmpeg. The Results Browser tab caches these re-encodes so you only pay the cost once.

---

## Requirements

```
opencv-python
torch
torchvision
numpy
pillow
tqdm
paddlepaddle
paddleocr
streamlit
```

Plus `ffmpeg` needs to be installed on the system for video re-encoding.

For training only: `wandb`, `roboflow`.
