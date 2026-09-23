# Week 08 — Real-Time Object Detection with YOLO

> **Difficulty:** Intermediate ★★★★  
> **Estimated duration:** 10–12 hours

This project follows the Week 08 curriculum and demonstrates how a YOLOv8 object-detection system works from pretrained inference through custom-data fine-tuning and lightweight deployment.

The project is intentionally organized so that **someone viewing the GitHub repository can understand the workflow without reading the source code first.**

---

## 1. What does this project do?

Instead of only answering **"What is in this image?"**, object detection answers:

> **What objects are present, where are they, and how confident is the model?**

YOLO produces a bounding box and confidence score for each detected object.

The project has two stages:

```text
PRETRAINED YOLO
      │
      ├── Image detection
      ├── Video detection
      └── Webcam + FPS
              │
              ▼
CUSTOM YOLO MODEL
      │
      ├── 3 custom classes
      ├── 50-epoch transfer learning
      ├── mAP@0.5 evaluation
      ├── 20 test images
      └── ONNX export + FPS
```

---

## 2. What is YOLO?

YOLO (You Only Look Once) is an object-detection architecture that predicts object locations and classes in a single model pipeline.

For every detection, the model provides:

- **Class** — what the object is
- **Confidence** — how confident the model is
- **Bounding box** — `(x1, y1, x2, y2)` coordinates

Example:

```text
person   0.94   (120, 80, 410, 600)
car      0.88   (500, 210, 790, 450)
```

This is different from image classification, where the model normally produces one class for the entire image.

---

## 3. Repository workflow

### Stage A — Pretrained model

The pretrained YOLOv8 model already knows common COCO objects.

Run:

```powershell
py -3.12 scripts/run_image_detection.py
```

This downloads/uses `yolov8n.pt`, detects objects in the example image, prints a clean detection table, and saves an annotated image.

### Stage B — Video

Provide a traffic/video file:

```powershell
py -3.12 scripts/run_video_detection.py --source data/input_video.mp4
```

The script:

1. Reads each frame.
2. Runs YOLO.
3. Counts `person`, `car`, and `bus`.
4. Saves an annotated video.
5. Saves frame-by-frame counts.
6. Creates a count-over-time plot.

### Stage C — Webcam

Run:

```powershell
py -3.12 scripts/webcam_detection.py --model yolov8n.pt
```

For the larger model:

```powershell
py -3.12 scripts/webcam_detection.py --model yolov8m.pt
```

Press **Q** to stop.

The displayed FPS demonstrates the speed/accuracy trade-off between the smaller and larger pretrained models.

---

# 4. Custom dataset

The curriculum requires three custom classes and 100 images per class.

Example:

```text
classes:
  0: pen
  1: bottle
  2: phone
```

Your own three classes can be used instead.

Recommended dataset structure:

```text
data/
└── custom_dataset/
    ├── images/
    │   ├── train/
    │   ├── val/
    │   └── test/
    └── labels/
        ├── train/
        ├── val/
        └── test/
```

Each image has a matching YOLO label file.

Example:

```text
images/train/image001.jpg
labels/train/image001.txt
```

Label format:

```text
class_id center_x center_y width height
```

Coordinates are normalized between 0 and 1.

Use Roboflow or LabelImg to annotate/export the dataset in YOLO format.

See `data/README.md`.

---

# 5. Fine-tuning

After preparing the dataset and `custom_dataset.yaml`:

```powershell
py -3.12 scripts/train_custom.py
```

The training configuration follows the curriculum:

```text
Model:      YOLOv8n
Epochs:     50
Image size: 640
Batch:      16
Transfer learning: pretrained weights
```

The training run produces loss and validation metrics.

After training, copy the best checkpoint:

```text
runs/.../weights/best.pt
```

to:

```text
models/best.pt
```

Do not commit large model files unless you intentionally want to distribute the trained model.

---

# 6. Evaluation

Run:

```powershell
py -3.12 scripts/evaluate_custom.py
```

The evaluation reports:

- mAP@0.5
- mAP@0.5:0.95
- validation/test plots
- predictions on up to 20 test images

The project does **not** hard-code a fake accuracy or mAP value. The actual result depends on your custom images, labels, classes, split, hardware, and training run.

---

# 7. ONNX deployment

Export the trained model:

```powershell
py -3.12 scripts/export_onnx.py
```

The script:

1. Loads `models/best.pt`.
2. Exports an ONNX model.
3. Measures inference time.
4. Calculates approximate FPS.

This demonstrates the deployment part of the curriculum.

---

# 8. Understanding the important metrics

### Confidence

Confidence represents how strongly the detector supports a particular detection.

Example:

```text
bottle → 0.92
```

### IoU — Intersection over Union

IoU compares the predicted bounding box with the ground-truth bounding box.

```text
IoU = Area of overlap / Area of union
```

Higher IoU means the predicted box overlaps the correct box more closely.

### NMS — Non-Maximum Suppression

A detector can produce multiple overlapping boxes for the same object.

NMS keeps the stronger detection and suppresses redundant overlapping detections.

### mAP@0.5

Mean Average Precision at IoU threshold 0.5 is a standard object-detection evaluation metric.

The repository reports the actual value produced by your trained model.

### FPS

Frames per second measures inference speed.

Higher FPS generally means faster real-time processing, while model size and accuracy can affect speed.

---

# 9. Installation

This project uses **Python 3.12**.

Open PowerShell in the repository folder:

```powershell
py -3.12 -m pip install --upgrade pip
py -3.12 -m pip install -r requirements.txt
```

No virtual environment is required for the workflow documented here.

Verify Ultralytics:

```powershell
py -3.12 -c "from ultralytics import YOLO; print('Ultralytics OK')"
```

---

# 10. Recommended learning order

If you are presenting this project in an interview, follow this order:

```text
1. Image detection
       ↓
2. Understand boxes + confidence
       ↓
3. Video detection
       ↓
4. Object counting
       ↓
5. Webcam + FPS
       ↓
6. Create/label custom dataset
       ↓
7. Fine-tune YOLOv8n
       ↓
8. Evaluate mAP
       ↓
9. Test on 20 images
       ↓
10. Export ONNX + measure FPS
```

This makes the repository demonstrate the complete object-detection lifecycle rather than just a single YOLO prediction command.

---

# 11. Repository structure

```text
Week_08_Real-Time_Object_Detection_with_YOLO/
│
├── README.md
├── requirements.txt
├── .gitignore
│
├── scripts/
│   ├── run_image_detection.py
│   ├── run_video_detection.py
│   ├── webcam_detection.py
│   ├── train_custom.py
│   ├── evaluate_custom.py
│   └── export_onnx.py
│
├── data/
│   ├── README.md
│   ├── custom_dataset.yaml
│   ├── custom_dataset/
│   └── test_images/
│
├── models/
│   └── best.pt
│
└── outputs/
    ├── README.md
    ├── image_detection/
    ├── video/
    ├── training/
    └── custom_predictions/
```

Dataset/model/output files are ignored by Git where appropriate.

---

# 12. How someone else can understand the project

A person visiting this repository should start with:

**README → data/README.md → scripts → outputs**

The README explains the concept and pipeline first. The scripts then correspond directly to stages of the project.

For example:

| Script | Purpose |
|---|---|
| `run_image_detection.py` | First YOLO inference |
| `run_video_detection.py` | Video detection + object counts |
| `webcam_detection.py` | Real-time webcam + FPS |
| `train_custom.py` | Fine-tune YOLOv8n |
| `evaluate_custom.py` | mAP + test predictions |
| `export_onnx.py` | ONNX deployment + FPS |

This separation makes the repository easier to understand and present.

---

## GitHub repository name

`Week_08_Real-Time_Object_Detection_with_YOLO`

## GitHub description

`Real-time object detection with YOLOv8 using image, video, webcam, custom dataset fine-tuning, mAP evaluation, and ONNX deployment.`

