import json
import os

# 입력 JSON 파일 경로와 출력 디렉토리 설정
input_annotation_file = "./ds2_dense/deepscores_train.json"  # 원본 JSON 파일
output_yolo_dir = "./yolo_labels"  # YOLO 형식 라벨 저장 디렉토리
os.makedirs(output_yolo_dir, exist_ok=True)

# JSON 파일 읽기
with open(input_annotation_file, 'r') as f:
    data = json.load(f)

# YOLO 포맷 변환 함수
def convert_to_yolo_format(bbox, img_width, img_height):
    x, y, w, h = bbox
    x_center = (x + w / 2) / img_width
    y_center = (y + h / 2) / img_height
    w /= img_width
    h /= img_height
    return x_center, y_center, w, h

# 이미지별 라벨 파일 생성
for image_info in data['images']:
    image_id = image_info['id']
    filename = image_info['filename']  # 'file_name' 대신 'filename'
    width, height = image_info['width'], image_info['height']

    # YOLO 라벨 파일 생성
    label_file = os.path.join(output_yolo_dir, f"{os.path.splitext(filename)[0]}.txt")
    with open(label_file, 'w') as f_out:
        for annotation in data['annotations']:
            if annotation['image_id'] == image_id:  # 이미지 ID와 매칭
                class_id = annotation['category_id'] - 1  # 클래스 ID는 0부터 시작
                bbox = annotation['bbox']  # [x, y, width, height]
                x_center, y_center, w, h = convert_to_yolo_format(bbox, width, height)
                f_out.write(f"{class_id} {x_center} {y_center} {w} {h}\n")

print(f"YOLO 라벨 파일이 {output_yolo_dir}에 저장되었습니다.")
