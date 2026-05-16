import pandas as pd

df = pd.read_csv("./test/tracking_results_botsort.csv")

line_x = 200
prev_side = {}

total_cross_count = 0
class_cross_count = {} 

for idx, row in df.iterrows():
    frame = row['Frame']
    track_id = row['Tracking_ID']
    class_name = row['Class_Name']
    x1, y1, x2, y2 = row['x1'], row['y1'], row['x2'], row['y2']
    
    center_x = (x1 + x2) / 2.0
    
    if center_x < line_x:
        current_side = -1
    else:
        current_side = 1
    
    if track_id in prev_side:
        prev_side_val = prev_side[track_id]
        if prev_side_val != current_side:
            total_cross_count += 1
            class_cross_count[class_name] = class_cross_count.get(class_name, 0) + 1
    prev_side[track_id] = current_side

print(f"总计跨越物体次数: {total_cross_count}")
print("各类别跨越次数:")
for cls, cnt in class_cross_count.items():
    print(f"  {cls}: {cnt}")