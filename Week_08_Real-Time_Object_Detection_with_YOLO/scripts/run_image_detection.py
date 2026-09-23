from pathlib import Path
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
SOURCE = "https://ultralytics.com/images/bus.jpg"
OUTPUT = ROOT / "outputs" / "image_detection"

model = YOLO("yolov8n.pt")
results = model.predict(source=SOURCE, save=True, project=OUTPUT, name="bus")

result = results[0]
names = result.names

print("\nDetected objects")
print("-" * 65)
print(f"{'Class':<20}{'Confidence':<15}{'x1':<10}{'y1':<10}{'x2':<10}{'y2':<10}")

for box in result.boxes:
    cls_id = int(box.cls.item())
    conf = float(box.conf.item())
    x1, y1, x2, y2 = box.xyxy[0].tolist()
    print(f"{names[cls_id]:<20}{conf:<15.3f}{x1:<10.1f}{y1:<10.1f}{x2:<10.1f}{y2:<10.1f}")

print(f"\nAnnotated output saved to: {OUTPUT / 'bus'}")
