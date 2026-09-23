from pathlib import Path
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
DATA_YAML = ROOT / "data" / "custom_dataset.yaml"
RUNS = ROOT / "outputs" / "training"

if not DATA_YAML.exists():
    raise FileNotFoundError(
        "data/custom_dataset.yaml not found. Follow data/README.md first."
    )

model = YOLO("yolov8n.pt")

results = model.train(
    data=str(DATA_YAML),
    epochs=50,
    imgsz=640,
    batch=16,
    project=str(RUNS),
    name="yolov8n_custom",
    pretrained=True,
)

best = Path(results.save_dir) / "weights" / "best.pt"
print(f"\nTraining complete.")
print(f"Best model: {best}")
print("Training curves are available in the training run folder.")
