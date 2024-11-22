import os
import shutil
from sklearn.model_selection import train_test_split
from pathlib import Path

# 디렉토리 설정
image_dir = "./ds2_dense/images"  # 이미지 파일 디렉토리
label_dir = "./yolo_labels"       # YOLO 형식 라벨 디렉토리
output_dir = "./yolo_dataset"     # YOLO 학습 데이터 디렉토리

train_image_dir = os.path.join(output_dir, "train/images")
train_label_dir = os.path.join(output_dir, "train/labels")
val_image_dir = os.path.join(output_dir, "val/images")
val_label_dir = os.path.join(output_dir, "val/labels")

# 디렉토리 생성
os.makedirs(train_image_dir, exist_ok=True)
os.makedirs(train_label_dir, exist_ok=True)
os.makedirs(val_image_dir, exist_ok=True)
os.makedirs(val_label_dir, exist_ok=True)

# 이미지 파일과 라벨 파일 매칭
image_files = list(Path(image_dir).glob("*.png"))
label_files = [os.path.join(label_dir, f"{img.stem}.txt") for img in image_files]

# 데이터 분할 (80% 학습, 20% 검증)
train_images, val_images, train_labels, val_labels = train_test_split(
    image_files, label_files, test_size=0.2, random_state=42
)

# 파일 복사
for img, lbl in zip(train_images, train_labels):
    if os.path.exists(lbl):  # 라벨 파일 존재 여부 확인
        shutil.copy(img, train_image_dir)
        shutil.copy(lbl, train_label_dir)
    else:
        print(f"라벨 파일 없음: {lbl}")

for img, lbl in zip(val_images, val_labels):
    if os.path.exists(lbl):  # 라벨 파일 존재 여부 확인
        shutil.copy(img, val_image_dir)
        shutil.copy(lbl, val_label_dir)
    else:
        print(f"라벨 파일 없음: {lbl}")

print(f"YOLO 학습 데이터가 {output_dir}에 준비되었습니다.")
