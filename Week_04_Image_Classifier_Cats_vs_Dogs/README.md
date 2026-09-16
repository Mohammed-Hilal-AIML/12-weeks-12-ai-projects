# Week 04 — Image Classifier: Cats vs. Dogs 🐱🐶

A beginner-friendly **Convolutional Neural Network (CNN)** project that classifies images as either **Cat** or **Dog** using **TensorFlow/Keras**.

## 📌 Project Overview

In this project, we build a CNN from scratch and train it on the **Dogs vs. Cats** dataset.

The project covers:

* Neural network basics
* Convolution and pooling
* Image resizing and normalization
* Image augmentation
* Training with epochs and batches
* Detecting overfitting
* Saving the best Keras model
* Evaluating model accuracy
* Predicting custom images

The target is approximately **85%+ accuracy**.

---

## 🐍 Python Requirement

> **Python 3.12 is required/recommended for this project.**

Check your Python version:

```bash
python --version
```

You should have:

```text
Python 3.12.x
```

If multiple Python versions are installed:

```bash
py --list
```

Run this project with Python 3.12:

```bash
py -3.12 cats_vs_dogs.py
```

---

## 📦 Requirements

Install the dependencies:

```bash
py -3.12 -m pip install -r requirements.txt
```

Main technologies:

* Python 3.12
* TensorFlow 2.x
* Keras
* NumPy
* Matplotlib
* Kaggle API

---

## 📊 Dataset

This project uses the **Dogs vs. Cats** dataset from Kaggle.

The original dataset contains approximately **25,000 images**.

Kaggle dataset:

[https://www.kaggle.com/competitions/dogs-vs-cats/data](https://www.kaggle.com/competitions/dogs-vs-cats/data)

You need a free Kaggle account to access the dataset.

---

## 📂 Dataset Structure

The images should be separated into **80% training** and **20% validation**.

```text
data/
├── train/
│   ├── cats/
│   └── dogs/
│
└── val/
    ├── cats/
    └── dogs/
```

**Important:** Do not put the same image in both `train` and `val`.

The validation images should be different from the training images.

---

## 🗂️ Project Structure

```text
Week_04_Image_Classifier_Cats_vs_Dogs/
│
├── cats_vs_dogs.py
├── README.md
├── requirements.txt
├── .gitignore
│
├── data/
│   ├── train/
│   │   ├── cats/
│   │   └── dogs/
│   └── val/
│       ├── cats/
│       └── dogs/
│
├── custom_images/
│
├── models/
│
└── outputs/
```

---

## 🔄 Image Preprocessing & Augmentation

Training images use:

```python
rescale=1./255
horizontal_flip=True
zoom_range=0.2
rotation_range=20
```

This helps the model generalize to different images.

Validation images are rescaled but are not augmented.

---

## 🧠 CNN Architecture

The model follows the Week 04 architecture:

```text
Conv2D(32)
     ↓
MaxPooling2D
     ↓
Conv2D(64)
     ↓
MaxPooling2D
     ↓
Conv2D(128)
     ↓
MaxPooling2D
     ↓
Flatten
     ↓
Dense(512)
     ↓
Dropout(0.5)
     ↓
Dense(1, sigmoid)
```

The model uses:

```text
Optimizer: Adam
Loss: Binary Crossentropy
Output: Sigmoid
```

---

## 🚀 Installation

### 1. Check Python

```bash
py -3.12 --version
```

Expected:

```text
Python 3.12.x
```

### 2. Upgrade pip

```bash
py -3.12 -m pip install --upgrade pip
```

### 3. Install dependencies

```bash
py -3.12 -m pip install -r requirements.txt
```

### 4. Verify TensorFlow

```bash
py -3.12 -c "import tensorflow as tf; print(tf.__version__)"
```

---

## ▶️ Run the Project

From the project directory:

```bash
py -3.12 cats_vs_dogs.py
```

The program will:

1. Check TensorFlow
2. Check available GPU devices
3. Check the dataset
4. Count training and validation images
5. Create image generators
6. Build the CNN
7. Train for 20 epochs
8. Save the best model
9. Generate accuracy and loss graphs
10. Evaluate the model
11. Generate a 3×3 prediction grid
12. Test custom images

---

## 💾 Saved Model

The best model is saved as:

```text
models/cats_vs_dogs_best.keras
```

---

## 📈 Output Files

After training, the project generates:

```text
outputs/
├── accuracy_curve.png
├── loss_curve.png
└── test_predictions_3x3.png
```

### Accuracy Curve

Shows training and validation accuracy.

### Loss Curve

Shows training and validation loss.

These curves can be used to observe possible **overfitting**.

### Prediction Grid

The project creates a **3×3 grid** showing predictions and true labels.

Incorrect predictions are highlighted in red.

---

## 🖼️ Custom Images

Place up to **5 custom pet images** inside:

```text
custom_images/
```

Supported formats:

```text
.jpg
.jpeg
.png
.bmp
.webp
```

Example:

```text
custom_images/
├── pet1.jpg
├── pet2.png
├── pet3.webp
├── pet4.jpg
└── pet5.png
```

The model will display the prediction probability and predicted class.

---

## 🐺 Edge Case Testing

The project can also be tested with images of:

* Wolves
* Foxes

These are outside the original cat/dog training classes and can be used to observe how the binary classifier responds to unfamiliar images.

---

## ⚠️ GitHub Note

Do **not** upload the complete dataset to GitHub.

The `.gitignore` file excludes:

```text
data/
custom_images/
models/
outputs/
```

This keeps the GitHub repository small.

---

## ✅ Week 04 Checklist

* [x] TensorFlow 2.x
* [x] Python 3.12
* [x] Dogs vs. Cats dataset
* [x] 80/20 train-validation split
* [x] ImageDataGenerator
* [x] Image augmentation
* [x] CNN from scratch
* [x] 20 training epochs
* [x] Best model checkpoint
* [x] Accuracy curve
* [x] Loss curve
* [x] Model evaluation
* [x] 3×3 prediction grid
* [x] Custom image testing
* [x] Wolf/fox edge-case testing

---

## 🛠️ Technologies

```text
Python 3.12
TensorFlow 2.x
Keras
NumPy
Matplotlib
Kaggle API
```

---

## 👨‍💻 Project

**Week 04 — Image Classifier: Cats vs. Dogs**

Part of a **12-Week Beginner AI Project Curriculum**.
