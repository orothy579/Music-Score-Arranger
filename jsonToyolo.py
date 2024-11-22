import json
import os

# JSON 파일 경로
input_annotation_file = "./ds2_dense/deepscores_train.json"
output_label_dir = "./yolo_labels"
os.makedirs(output_label_dir, exist_ok=True)

# JSON 파일 읽기
with open(input_annotation_file, 'r') as f:
    data = json.load(f)

# 어노테이션 데이터
annotations = data['annotations']

# 라벨 파일 생성
for image_info in data['images']:
    filename = image_info['filename']
    width, height = image_info['width'], image_info['height']
    ann_ids = image_info.get('ann_ids', [])

    # 라벨 파일 경로
    label_file = os.path.join(output_label_dir, f"{os.path.splitext(filename)[0]}.txt")

    # 해당 이미지에 대한 어노테이션 처리
    lines = []
    for ann_id in ann_ids:
        if ann_id in annotations:
            ann = annotations[ann_id]
            bbox = ann.get('a_bbox')  # [x_min, y_min, x_max, y_max]
            cat_ids = ann.get('cat_id', [])  # ['135', '208']
            
            # 바운딩 박스와 클래스 ID 확인
            if bbox and cat_ids:
                # YOLO에서는 하나의 클래스 ID만 사용 가능 (첫 번째 ID 선택)
                class_id = int(cat_ids[0]) - 1  # 클래스 ID는 0부터 시작
                x_min, y_min, x_max, y_max = bbox
                x_center = (x_min + x_max) / 2 / width
                y_center = (y_min + y_max) / 2 / height
                w = (x_max - x_min) / width
                h = (y_max - y_min) / height

                # YOLO 형식 라벨 추가
                lines.append(f"{class_id} {x_center} {y_center} {w} {h}")

    # 라벨 파일에 작성
    with open(label_file, 'w') as f_out:
        f_out.write("\n".join(lines))

print(f"라벨 파일이 {output_label_dir}에 생성되었습니다.")
