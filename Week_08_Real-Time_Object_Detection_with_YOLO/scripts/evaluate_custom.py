from pathlib import Path
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
DATA_YAML = ROOT / "data" / "custom_dataset.yaml"
MODEL = ROOT / "models" / "best.pt"
TEST_DIR = ROOT / "data" / "test_images"
OUTPUT = ROOT / "outputs" / "custom_predictions"

if not MODEL.exists():
    raise FileNotFoundError(
        "models/best.pt not found. Copy the trained best.pt there first."
    )

if not DATA_YAML.exists():
    raise FileNotFoundError("data/custom_dataset.yaml not found.")

model = YOLO(str(MODEL))

metrics = model.val(
    data=str(DATA_YAML),
    imgsz=640,
    split="test",
    plots=True,
    project=str(OUTPUT),
    name="evaluation",
)

print("\nCustom model evaluation")
print(f"mAP@0.5: {metrics.box.map50:.4f}")
print(f"mAP@0.5:0.95: {metrics.box.map:.4f}")

if TEST_DIR.exists():
    image_files = [
        p for p in TEST_DIR.iterdir()
        if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
    ][:20]

    if image_files:
        model.predict(
            source=[str(p) for p in image_files],
            save=True,
            project=str(OUTPUT),
            name="test_predictions",
            imgsz=640,
        )
        print(f"Predictions generated for {len(image_files)} test images.")
    else:
        print("No test images found in data/test_images.")
