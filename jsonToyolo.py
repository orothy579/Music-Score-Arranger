import json
import os

# JSON 파일 경로 설정
input_annotation_file = "./ds2_dense/deepscores_train.json"
output_yolo_dir = "./yolo_labels"

# 출력 디렉토리 생성
os.makedirs(output_yolo_dir, exist_ok=True)

# JSON 파일 읽기
with open(input_annotation_file, 'r') as f:
    data = json.load(f)

# 'annotations'가 딕셔너리라면 리스트로 변환
if isinstance(data['annotations'], dict):
    annotations = list(data['annotations'].values())
else:
    annotations = data['annotations']

# 'annotations' 항목이 문자열이라면 JSON 객체로 변환
if isinstance(annotations[0], str):
    annotations = [json.loads(ann) for ann in annotations]

# 'image_id' 키 동적으로 설정
img_id_key = 'image_id' if 'image_id' in annotations[0] else 'img_id'

# YOLO 포맷 변환 함수
def convert_to_yolo_format(bbox, img_width, img_height):
    x, y, w, h = bbox
    x_center = (x + w / 2) / img_width
    y_center = (y + h / 2) / img_height
    w /= img_width
    h /= img_height
    return x_center, y_center, w, h

# YOLO 라벨 생성
for image_info in data['images']:
    image_id = image_info['id']
    filename = image_info.get('filename', image_info.get('file_name'))  # 'filename' 또는 'file_name'
    width, height = image_info['width'], image_info['height']

    # YOLO 라벨 파일 경로
    label_file = os.path.join(output_yolo_dir, f"{os.path.splitext(filename)[0]}.txt")
    
    # YOLO 라벨 작성
    with open(label_file, 'w') as f_out:
        for annotation in annotations:
            if annotation[img_id_key] == image_id:  # 이미지 ID 매칭
                class_id = annotation['category_id'] - 1  # 클래스 ID는 0부터 시작
                bbox = annotation['bbox']  # [x, y, width, height]
                x_center, y_center, w, h = convert_to_yolo_format(bbox, width, height)
                f_out.write(f"{class_id} {x_center} {y_center} {w} {h}\n")

print(f"YOLO 라벨 파일이 {output_yolo_dir}에 저장되었습니다.")
