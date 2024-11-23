import os

# 라벨 디렉토리 설정
label_dir = "/Users/lch/development/opencv/finalProject/yolo_dataset/train/labels"
nc = 185  # data.yaml에 설정한 클래스 수

# 클래스 ID 점검
problem_files = []
for label_file in os.listdir(label_dir):
    label_path = os.path.join(label_dir, label_file)
    with open(label_path, "r") as f:
        for line in f:
            class_id = int(line.strip().split()[0])  # 클래스 ID 추출
            if not (0 <= class_id < nc):
                problem_files.append((label_file, class_id))

# 문제 있는 파일 출력
if problem_files:
    print("잘못된 클래스 ID가 포함된 라벨 파일:")
    for file, class_id in problem_files:
        print(f"{file}: 클래스 ID {class_id}")
else:
    print("모든 라벨 파일이 올바릅니다.")
