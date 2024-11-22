import os

# YOLO 데이터셋 디렉토리 경로 설정
yolo_dataset_dir = "./yolo_dataset"

# 필요한 디렉토리 생성
os.makedirs(os.path.join(yolo_dataset_dir, "train/images"), exist_ok=True)
os.makedirs(os.path.join(yolo_dataset_dir, "train/labels"), exist_ok=True)
os.makedirs(os.path.join(yolo_dataset_dir, "val/images"), exist_ok=True)
os.makedirs(os.path.join(yolo_dataset_dir, "val/labels"), exist_ok=True)

print(f"YOLO 데이터셋 디렉토리가 {yolo_dataset_dir}에 생성되었습니다.")
