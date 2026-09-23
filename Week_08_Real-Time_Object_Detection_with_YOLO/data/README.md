# Custom Dataset Setup

The Week 08 manual requires a custom **3-class dataset with 100 images per class**, labeled using Roboflow or LabelImg and exported in YOLO format.

Example classes:

- `pen`
- `bottle`
- `phone`

You may choose your own three classes.

## Recommended structure

```text
data/custom_dataset/
├── images/
│   ├── train/
│   ├── val/
│   └── test/
└── labels/
    ├── train/
    ├── val/
    └── test/
```

Each image needs a corresponding `.txt` label.

Example:

```text
images/train/phone_001.jpg
labels/train/phone_001.txt
```

## YOLO label format

Each line is:

```text
class_id center_x center_y width height
```

Coordinates are normalized from 0 to 1.

## YAML

Create:

```text
data/custom_dataset.yaml
```

Example:

```yaml
path: data/custom_dataset
train: images/train
val: images/val
test: images/test

names:
  0: pen
  1: bottle
  2: phone
```

If you choose different classes, change `names`.

## Labeling

Use Roboflow or LabelImg to draw bounding boxes around each object and export in YOLO format.

The goal from the manual is approximately:

```text
3 classes × 100 images = 300 images
```

Keep a separate test set so the final evaluation is performed on images the model did not train on.
