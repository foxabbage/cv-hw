import cv2
import pandas as pd
import os
from pathlib import Path

def visualize_frames_simple(video_path, csv_path, start_frame=50, end_frame=70, output_dir='output_frames'):
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(csv_path)
    cap = cv2.VideoCapture(video_path)
    
    for frame_num in range(start_frame, end_frame + 1):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = cap.read()
        if not ret:
            print(f"{frame_num} frame not available")
            continue
        frame_detections = df[df['Frame'] == frame_num]
        img_with_bbox = frame.copy()
        for _, det in frame_detections.iterrows():
            x1, y1, x2, y2 = int(det['x1']), int(det['y1']), int(det['x2']), int(det['y2'])
            tracking_id = int(det['Tracking_ID'])
            class_name = det['Class_Name']
            confidence = det['Confidence']
            cv2.rectangle(img_with_bbox, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"ID:{tracking_id} {class_name} {confidence:.2f}"
            cv2.putText(img_with_bbox, label, (x1, y1-5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.putText(img_with_bbox, f"Frame: {frame_num}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        output_path = os.path.join(output_dir, f'frame_{frame_num:04d}.png')
        cv2.imwrite(output_path, img_with_bbox)
    cap.release()

if __name__ == "__main__":
    visualize_frames_simple(
        video_path="./test/test.mp4",
        csv_path="./test/tracking_results_botsort.csv",
        start_frame=91,
        end_frame=110,
        output_dir="visualization_frames_botsort"
    )