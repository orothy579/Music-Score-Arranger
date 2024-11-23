import os

# 라벨 디렉토리 설정
label_dir = "./yolo_dataset/train/labels"

# 모든 라벨 파일 수정
for label_file in os.listdir(label_dir):
    label_path = os.path.join(label_dir, label_file)
    with open(label_path, "r") as f:
        lines = f.readlines()

    # 수정된 라벨 저장
    updated_lines = []
    for line in lines:
        class_id, *coords = line.strip().split()
        class_id = int(class_id) + 1  # 클래스 ID를 1 감소
        updated_lines.append(f"{class_id} {' '.join(coords)}\n")

    # 파일 덮어쓰기
    with open(label_path, "w") as f:
        f.writelines(updated_lines)

print("클래스 ID 조정 완료!")
