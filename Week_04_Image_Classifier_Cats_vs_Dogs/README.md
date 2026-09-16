# Week 04 — Image Classifier: Cats vs. Dogs

Beginner AI Project Curriculum — Week 4 of 12.

## Project

Build a Convolutional Neural Network (CNN) from scratch using TensorFlow/Keras to distinguish cats from dogs.

The project follows the Week 04 curriculum requirements:
- TensorFlow 2.x / Keras
- Dogs vs. Cats Kaggle dataset
- Train/validation folder structure
- 80/20 dataset split
- Image preprocessing and augmentation
- CNN with the specified architecture
- 20 training epochs
- ModelCheckpoint for the best model
- Accuracy and loss curves
- 3×3 test prediction grid
- Custom image predictions and probabilities
- Edge-case testing such as a wolf or fox

## Folder Structure

```text
Week_04_Image_Classifier_Cats_vs_Dogs/
├── cats_vs_dogs.py
├── README.md
├── requirements.txt
├── .gitignore
├── data/
│   ├── train/
│   │   ├── cats/
│   │   └── dogs/
│   └── val/
│       ├── cats/
│       └── dogs/
├── custom_images/
├── models/
└── outputs/
```

> The `data/`, `custom_images/`, `models/`, and `outputs/` directories can be empty initially. Generated models and outputs are ignored by Git.

## Dataset

Download the **Dogs vs. Cats** dataset from Kaggle. The curriculum specifies 25,000 images.

Organize the images into an 80/20 train/validation split:

```text
data/
├── train/
│   ├── cats/
│   └── dogs/
└── val/
    ├── cats/
    └── dogs/
```

Make sure the files are distributed between the four folders before running the program.

## Installation

Create and activate a virtual environment if desired:

```bash
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Run

```bash
python cats_vs_dogs.py
```

The script checks TensorFlow and available GPUs, counts dataset files, creates the image pipeline, builds the CNN, trains for 20 epochs, saves the best model, evaluates it, and runs custom-image predictions.

## CNN Architecture

The curriculum specifies:

```text
Conv2D(32)
    ↓
MaxPool
    ↓
Conv2D(64)
    ↓
MaxPool
    ↓
Conv2D(128)
    ↓
MaxPool
    ↓
Flatten
    ↓
Dense(512)
    ↓
Dropout(0.5)
    ↓
Dense(1, sigmoid)
```

The model uses the Adam optimizer.

## Image Pipeline

Training images use:

- `rescale=1./255`
- `horizontal_flip=True`
- `zoom_range=0.2`
- `rotation_range=20`

Validation images are rescaled to the `[0, 1]` range.

## Outputs

After training, the project creates:

```text
models/
└── cats_vs_dogs_best.keras

outputs/
├── accuracy_curve.png
├── loss_curve.png
└── test_predictions_3x3.png
```

The 3×3 prediction grid displays predicted and true labels, with incorrect predictions highlighted in red.

## Custom Images

Place up to five `.jpg`, `.jpeg`, `.png`, `.bmp`, or `.webp` images inside:

```text
custom_images/
```

The script prints the predicted class and dog probability for each image.

For the curriculum's edge-case experiment, you can also place a wolf or fox image in this folder and observe the model's prediction probability. These are out-of-distribution examples, so their predictions should be treated as an experiment rather than as reliable animal identification.

## Overfitting

The script saves separate accuracy and loss curves for training and validation. Compare the curves to observe where validation performance stops improving while training performance continues improving.

## Learning Goals

This project practices:
- Neural-network basics
- Convolution and pooling
- Image resizing and normalization
- Data augmentation
- Epochs and batch size
- Training/validation monitoring
- Overfitting detection
- Saving/loading a Keras model
- Image classification

## Requirements

Python 3.10+ is recommended. TensorFlow/Keras, NumPy, and Matplotlib are listed in `requirements.txt`.

## Source

Based on the provided **Week 04 · AI Project Curriculum Beginner — Image Classifier: Cats vs. Dogs** manual.
