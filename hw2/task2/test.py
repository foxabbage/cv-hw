import cv2
import csv
from ultralytics import YOLO
import os

MODEL_PATH = "./models/weights/best.pt"
VIDEO_PATH = "./test/test.mp4"
CSV_OUTPUT = "./test/tracking_results.csv"
SHOW_VIDEO = False

model = YOLO(MODEL_PATH)

cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    raise FileNotFoundError

with open(CSV_OUTPUT, mode='w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(["Frame", "Tracking_ID", "Class_ID", "Class_Name", "x1", "y1", "x2", "y2", "Confidence"])

frame_idx = 0
print("infer...")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = model.track(frame, persist=True, conf=0.4, iou=0.45, tracker="bytetrack.yaml", verbose=False)

    if results and results[0].boxes.id is not None:
        boxes = results[0].boxes.xyxy.cpu().numpy()
        classes = results[0].boxes.cls.cpu().numpy()
        track_ids = results[0].boxes.id.cpu().numpy().astype(int)
        confs = results[0].boxes.conf.cpu().numpy()

        for box, cls_id, tid, conf in zip(boxes, classes, track_ids, confs):
            x1, y1, x2, y2 = map(int, box)
            class_name = model.names[int(cls_id)]

            with open(CSV_OUTPUT, mode='a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow([frame_idx, tid, int(cls_id), class_name, x1, y1, x2, y2, round(conf, 3)])

            if SHOW_VIDEO:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label = f"{class_name} ID:{tid} {conf:.2f}"
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    if SHOW_VIDEO:
        cv2.imshow("Multi-Object Tracking", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    frame_idx += 1

cap.release()
if SHOW_VIDEO:
    cv2.destroyAllWindows()
print(f"{frame_idx} frames, output{CSV_OUTPUT}")