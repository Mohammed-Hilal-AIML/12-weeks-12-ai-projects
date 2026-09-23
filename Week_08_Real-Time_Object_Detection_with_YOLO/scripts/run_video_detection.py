import argparse
from pathlib import Path
from collections import Counter

import cv2
import pandas as pd
import matplotlib.pyplot as plt
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "outputs" / "video"
OUTPUT.mkdir(parents=True, exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument("--source", required=True, help="Path to a video file")
parser.add_argument("--model", default="yolov8n.pt")
args = parser.parse_args()

model = YOLO(args.model)
cap = cv2.VideoCapture(args.source)

if not cap.isOpened():
    raise FileNotFoundError(f"Could not open video: {args.source}")

fps = cap.get(cv2.CAP_PROP_FPS) or 30
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

out_video = OUTPUT / "annotated_video.mp4"
writer = cv2.VideoWriter(
    str(out_video),
    cv2.VideoWriter_fourcc(*"mp4v"),
    fps,
    (width, height),
)

rows = []
frame_number = 0

while True:
    ok, frame = cap.read()
    if not ok:
        break

    result = model.predict(frame, verbose=False)[0]
    annotated = result.plot()
    writer.write(annotated)

    counts = Counter()
    if result.boxes is not None:
        for cls_id in result.boxes.cls.tolist():
            counts[result.names[int(cls_id)]] += 1

    rows.append({
        "frame": frame_number,
        "time_seconds": frame_number / fps,
        "people": counts.get("person", 0),
        "cars": counts.get("car", 0),
        "buses": counts.get("bus", 0),
    })
    frame_number += 1

cap.release()
writer.release()

df = pd.DataFrame(rows)
csv_path = OUTPUT / "object_counts_over_time.csv"
df.to_csv(csv_path, index=False)

plt.figure(figsize=(10, 5))
for column in ["people", "cars", "buses"]:
    if column in df:
        plt.plot(df["time_seconds"], df[column], label=column)
plt.xlabel("Time (seconds)")
plt.ylabel("Objects detected")
plt.title("Object Counts Over Time")
plt.legend()
plt.tight_layout()
plt.savefig(OUTPUT / "counts_over_time.png", dpi=150)
plt.close()

print(f"Annotated video: {out_video}")
print(f"Counts CSV: {csv_path}")
print(f"Count plot: {OUTPUT / 'counts_over_time.png'}")
