"""
Week 04 - Image Classifier: Cats vs. Dogs
Beginner AI Project Curriculum

Build a CNN from scratch using TensorFlow/Keras.
"""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.callbacks import ModelCheckpoint

# -----------------------------
# Configuration
# -----------------------------
IMG_SIZE = (150, 150)
BATCH_SIZE = 32
EPOCHS = 20
DATA_DIR = Path("data")
TRAIN_DIR = DATA_DIR / "train"
VAL_DIR = DATA_DIR / "val"
MODEL_DIR = Path("models")
OUTPUT_DIR = Path("outputs")
CUSTOM_DIR = Path("custom_images")

MODEL_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

BEST_MODEL = MODEL_DIR / "cats_vs_dogs_best.keras"


def check_tensorflow():
    print("TensorFlow version:", tf.__version__)
    gpus = tf.config.list_physical_devices("GPU")
    print("GPU devices:", gpus if gpus else "None detected; training will use CPU.")


def check_dataset():
    required = [
        TRAIN_DIR / "cats",
        TRAIN_DIR / "dogs",
        VAL_DIR / "cats",
        VAL_DIR / "dogs",
    ]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "Dataset folders are missing:\n"
            + "\n".join(missing)
            + "\n\nCreate the required train/val folder structure and place the "
              "Cats vs. Dogs images inside it."
        )

    counts = {}
    for split in ("train", "val"):
        for label in ("cats", "dogs"):
            counts[f"{split}_{label}"] = len(
                list((DATA_DIR / split / label).glob("*"))
            )

    print("\nDataset file counts:")
    for name, count in counts.items():
        print(f"  {name}: {count}")

    return counts


def create_generators():
    # Curriculum-required preprocessing and augmentation.
    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        horizontal_flip=True,
        zoom_range=0.2,
        rotation_range=20,
    )

    val_datagen = ImageDataGenerator(rescale=1.0 / 255)

    train_generator = train_datagen.flow_from_directory(
        TRAIN_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="binary",
        shuffle=True,
    )

    val_generator = val_datagen.flow_from_directory(
        VAL_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="binary",
        shuffle=False,
    )

    return train_generator, val_generator


def build_cnn():
    # Curriculum-required architecture.
    model = Sequential([
        Conv2D(32, (3, 3), activation="relu", input_shape=(*IMG_SIZE, 3)),
        MaxPooling2D(),

        Conv2D(64, (3, 3), activation="relu"),
        MaxPooling2D(),

        Conv2D(128, (3, 3), activation="relu"),
        MaxPooling2D(),

        Flatten(),
        Dense(512, activation="relu"),
        Dropout(0.5),
        Dense(1, activation="sigmoid"),
    ])

    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )

    model.summary()
    return model


def plot_training_history(history):
    plt.figure(figsize=(8, 5))
    plt.plot(history.history["accuracy"], label="Train Accuracy")
    plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training and Validation Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "accuracy_curve.png", dpi=150)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(history.history["loss"], label="Train Loss")
    plt.plot(history.history["val_loss"], label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "loss_curve.png", dpi=150)
    plt.close()


def evaluate_model(model, val_generator):
    loss, accuracy = model.evaluate(val_generator, verbose=1)
    print(f"\nTest/validation accuracy: {accuracy:.4f}")
    print(f"Test/validation loss: {loss:.4f}")

    # 3x3 grid, as required by the curriculum.
    val_generator.reset()
    images, labels = next(val_generator)
    probabilities = model.predict(images, verbose=0).ravel()
    predictions = (probabilities >= 0.5).astype(int)

    class_names = {0: "cat", 1: "dog"}

    plt.figure(figsize=(10, 10))
    for i in range(min(9, len(images))):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(images[i])

        true_label = class_names[int(labels[i])]
        pred_label = class_names[int(predictions[i])]
        correct = true_label == pred_label

        title = f"Pred: {pred_label}\nTrue: {true_label}"
        ax.set_title(title, color="black" if correct else "red")
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "test_predictions_3x3.png", dpi=150)
    plt.close()


def predict_custom_images(model):
    if not CUSTOM_DIR.exists():
        print("\ncustom_images/ not found; skipping custom image predictions.")
        return

    image_files = [
        p for p in CUSTOM_DIR.iterdir()
        if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    ][:5]

    if not image_files:
        print("\nNo custom images found in custom_images/; skipping.")
        return

    print("\nCustom image predictions:")
    for path in image_files:
        image = load_img(path, target_size=IMG_SIZE)
        array = img_to_array(image) / 255.0
        probability = float(model.predict(np.expand_dims(array, axis=0), verbose=0)[0][0])

        label = "dog" if probability >= 0.5 else "cat"
        print(f"  {path.name}: {label} (dog probability: {probability:.4f})")


def main():
    check_tensorflow()
    check_dataset()

    train_generator, val_generator = create_generators()
    model = build_cnn()

    checkpoint = ModelCheckpoint(
        BEST_MODEL,
        monitor="val_accuracy",
        save_best_only=True,
        mode="max",
        verbose=1,
    )

    print("\nTraining for 20 epochs...")
    history = model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=EPOCHS,
        callbacks=[checkpoint],
    )

    plot_training_history(history)

    print("\nLoading best saved model...")
    model = tf.keras.models.load_model(BEST_MODEL)

    evaluate_model(model, val_generator)
    predict_custom_images(model)

    print("\nDone. Check the outputs/ and models/ folders.")


if __name__ == "__main__":
    main()
