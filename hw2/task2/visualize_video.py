import cv2
import pandas as pd
import os
import hashlib

def get_color(track_id):
    h = hashlib.md5(str(track_id).encode()).hexdigest()
    b = int(h[:2], 16)
    g = int(h[2:4], 16)
    r = int(h[4:6], 16)
    return (b, g, r)

def visualize_tracking(csv_path, video_path, output_path, fps=30):
    df = pd.read_csv(csv_path)

    unique_ids = df['Tracking_ID'].unique()
    color_map = {tid: get_color(tid) for tid in unique_ids}
    frame_groups = {frame: group for frame, group in df.groupby('Frame')}

    cap = cv2.VideoCapture(video_path)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    fps = fps if fps > 0 else video_fps
    if fps <= 0: fps = 30

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    frame_idx = 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx in frame_groups:
            detections = frame_groups[frame_idx]
            for _, det in detections.iterrows():
                x1, y1 = int(det['x1']), int(det['y1'])
                x2, y2 = int(det['x2']), int(det['y2'])

                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(width, x2), min(height, y2)
                if x1 >= x2 or y1 >= y2:
                    continue

                track_id = int(det['Tracking_ID'])
                class_name = str(det['Class_Name'])
                conf = float(det['Confidence'])
                color = color_map.get(track_id, (0, 255, 0))

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                label = f"{class_name} #{track_id} {conf:.2f}"
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale, thickness = 0.5, 1
                (label_w, label_h), baseline = cv2.getTextSize(label, font, font_scale, thickness)

                pad = 3
                cv2.rectangle(frame, 
                              (x1, y1 - label_h - pad*2), 
                              (x1 + label_w + pad*2, y1), 
                              color, -1)
                cv2.putText(frame, label, 
                            (x1 + pad, y1 - pad), 
                            font, font_scale, (0, 0, 0), thickness)

        out.write(frame)
        frame_idx += 1

        if frame_idx % 30 == 0 or frame_idx == total_frames:
            print(f"已处理: {frame_idx}/{total_frames} 帧")

    cap.release()
    out.release()

if __name__ == "__main__":
    CSV_FILE = "./test/tracking_results_botsort.csv"
    VIDEO_FILE = "./test/test.mp4"
    OUTPUT_FILE = "./test/output_visualized.mp4"
    TARGET_FPS = 30

    visualize_tracking(CSV_FILE, VIDEO_FILE, OUTPUT_FILE, TARGET_FPS)