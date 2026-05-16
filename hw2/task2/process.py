import os
from PIL import Image
import numpy as np
from multiprocessing import Pool, cpu_count
from functools import partial
import argparse

VISDRONE_CLASSES = [
    'pedestrian', 'people', 'bicycle', 'car', 'van', 
    'truck', 'tricycle', 'awning-tricycle', 'bus', 'motor'
]

def process_single_annotation(ann_file, img_dir, ann_dir, out_label_dir):
    img_name = ann_file.replace('.txt', '.jpg')
    img_path = os.path.join(img_dir, img_name)
    
    if not os.path.exists(img_path):
        return f"Warning: Image not found {img_path}"
    
    try:
        with Image.open(img_path) as img:
            w, h = img.size
    except Exception as e:
        return f"Warning: Could not read image {img_path} - {e}"
    
    ann_path = os.path.join(ann_dir, ann_file)
    out_txt_path = os.path.join(out_label_dir, ann_file)
    
    try:
        with open(ann_path, 'r') as f:
            lines = f.readlines()
    except Exception as e:
        return f"Warning: Could not read annotation {ann_path} - {e}"
    
    valid_lines = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        parts = line.split(',')
        if len(parts) < 8:
            continue
            
        try:
            data = list(map(float, parts))
        except ValueError:
            continue
            
        x, y, bw, bh, score, cls_id, truncation, occlusion = data[:8]
        
        if score < 1:
            continue
            
        yolo_cls = int(cls_id) - 1
        if yolo_cls < 0 or yolo_cls >= 10:
            continue
            
        cx = (x + bw / 2) / w
        cy = (y + bh / 2) / h
        nw = bw / w
        nh = bh / h
        
        cx = max(0.0, min(1.0, cx))
        cy = max(0.0, min(1.0, cy))
        nw = max(0.0, min(1.0, nw))
        nh = max(0.0, min(1.0, nh))
        
        valid_lines.append(f"{yolo_cls} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}\n")

    with open(out_txt_path, 'w') as fw:
        fw.writelines(valid_lines)
    
    return f"Processed: {ann_file} ({len(valid_lines)} objects)"

def convert_split_parallel(img_dir, ann_dir, out_label_dir, num_workers=None):
    os.makedirs(out_label_dir, exist_ok=True)
    
    ann_files = [f for f in os.listdir(ann_dir) if f.endswith('.txt')]
    print(f"Found {len(ann_files)} annotation files in {ann_dir}")
    
    if num_workers is None:
        num_workers = min(cpu_count(), len(ann_files))
    
    print(f"Using {num_workers} workers...")
    
    process_func = partial(
        process_single_annotation,
        img_dir=img_dir,
        ann_dir=ann_dir,
        out_label_dir=out_label_dir
    )
    
    with Pool(processes=num_workers) as pool:
        results = pool.map(process_func, ann_files)
    
    success_count = 0
    warning_count = 0
    for result in results:
        if result.startswith("Warning"):
            warning_count += 1
            print(result)
        else:
            success_count += 1
            if success_count % 500 == 0:
                print(f"Progress: {success_count}/{len(ann_files)}")
    
    print(f"Completed: {success_count} successful, {warning_count} warnings")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert VisDrone dataset to YOLO format')
    parser.add_argument('--workers', type=int, default=None, 
                       help='Number of worker processes (default: CPU count)')
    args = parser.parse_args()
    
    base = './data/archive'
    convert_split_parallel(
        f'{base}/train/VisDrone2019-DET-train/images',
        f'{base}/train/VisDrone2019-DET-train/annotations',
        './visdrone_yolo/train/labels',
        num_workers=args.workers
    )
    
    convert_split_parallel(
        f'{base}/val/VisDrone2019-DET-val/images',
        f'{base}/val/VisDrone2019-DET-val/annotations',
        './visdrone_yolo/val/labels',
        num_workers=args.workers
    )