import json

# JSON 파일 경로
annotation_file = "/Users/lch/development/opencv/finalProject/ds2_dense/deepscores_train.json"

# JSON 데이터 읽기
with open(annotation_file, 'r') as f:
    data = json.load(f)

# 클래스 이름 추출
categories = data['categories']
class_names = {int(class_id): details['name'] for class_id, details in categories.items()}
print("클래스 이름:", class_names)

# YOLOv5 data.yaml 포맷으로 저장
with open("data.yaml", 'w') as f:
    f.write(f"nc: {len(class_names)}\n")  # 총 클래스 수
    f.write("names:\n")
    
    # 클래스 ID 순서대로 저장
    for class_id in sorted(class_names.keys()):
        f.write(f"  - {class_names[class_id]}\n")

print("data.yaml 파일이 생성되었습니다!")
