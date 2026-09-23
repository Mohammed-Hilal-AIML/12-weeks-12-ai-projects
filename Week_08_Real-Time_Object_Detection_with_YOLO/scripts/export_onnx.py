from pathlib import Path
import time
import numpy as np
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "models" / "best.pt"

if not MODEL.exists():
    raise FileNotFoundError("models/best.pt not found.")

model = YOLO(str(MODEL))
exported = model.export(format="onnx", imgsz=640)

print(f"ONNX model exported to: {exported}")

# Warm-up and simple inference-speed measurement.
sample = np.zeros((640, 640, 3), dtype=np.uint8)

for _ in range(5):
    model.predict(sample, verbose=False)

times = []
for _ in range(20):
    start = time.perf_counter()
    model.predict(sample, verbose=False)
    times.append(time.perf_counter() - start)

avg_time = sum(times) / len(times)
print(f"Average inference time: {avg_time:.4f} seconds")
print(f"Approximate FPS: {1 / avg_time:.2f}")
