import argparse
import time
import cv2
from ultralytics import YOLO

parser = argparse.ArgumentParser()
parser.add_argument("--model", default="yolov8n.pt", help="yolov8n.pt or yolov8m.pt")
args = parser.parse_args()

model = YOLO(args.model)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise RuntimeError("Webcam could not be opened.")

fps_values = []
previous = time.perf_counter()

print("Press Q to quit webcam detection.")

while True:
    ok, frame = cap.read()
    if not ok:
        break

    result = model.predict(frame, verbose=False)[0]
    annotated = result.plot()

    now = time.perf_counter()
    fps = 1.0 / max(now - previous, 1e-9)
    previous = now
    fps_values.append(fps)

    cv2.putText(
        annotated,
        f"FPS: {fps:.1f} | Model: {args.model}",
        (15, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (0, 255, 0),
        2,
    )

    cv2.imshow("YOLO Real-Time Detection", annotated)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

if fps_values:
    print(f"Average FPS: {sum(fps_values) / len(fps_values):.2f}")
