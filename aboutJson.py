# import json

# # JSON 파일 경로
# json_file = "./ds2_dense/deepscores_train.json"

# # JSON 파일 읽기
# with open(json_file, 'r') as f:
#     data = json.load(f)

# # 데이터 타입 확인
# print(type(data))  # dict 또는 list

# JSON 데이터의 상위 키 확인
# print(data.keys())

# # JSON 파일의 'images' 항목의 첫 번째 데이터 확인
# print(data['images'][0])


# for ann in data['annotations']:
#     print(type(ann))  # 각 항목의 타입 확인
#     print(ann)        # 첫 번째 항목 내용 출력
#     break


# annotations의 첫 번째 항목 확인
# print(data['annotations'][0])

# print(type(data['annotations']))  # 딕셔너리인지 확인
# # print(data['annotations'].keys())  # 딕셔너리의 키 확인
# print(data['annotations']['412579'])  # 특정 키의 값 확인

# import json

# # JSON 파일 경로
# json_file = "./ds2_dense/deepscores_train.json"

# # JSON 데이터 로드
# with open(json_file, "r") as f:
#     data = json.load(f)

# import json

# # JSON 파일 경로
# json_file = "./ds2_dense/deepscores_train.json"

# # JSON 데이터 로드
# with open(json_file, "r") as f:
#     data = json.load(f)

# # 모든 클래스 ID 추출
# class_ids = set()
# for ann in data["annotations"].values():
#     cat_ids = ann.get("cat_id", [])
#     if cat_ids:  # cat_id가 None이 아니고 값이 있을 때만 처리
#         # cat_ids 내부의 None 값 제거
#         valid_ids = [int(cat_id) for cat_id in cat_ids if cat_id is not None]
#         class_ids.update(valid_ids)

# # 클래스 수 확인
# print(f"고유 클래스 수: {len(class_ids)}")
# print(f"클래스 ID 목록: {sorted(class_ids)}")


import json

# JSON 파일 경로
annotation_file = "/Users/lch/development/opencv/finalProject/ds2_dense/deepscores_train.json"

# JSON 데이터 읽기
with open(annotation_file, 'r') as f:
    data = json.load(f)

# JSON 데이터의 상위 키 확인
print("JSON 데이터의 상위 키들:", data.keys())

# 'categories' 키가 존재하는지 확인
if 'categories' in data:
    print("'categories' 키가 존재합니다.")
    print("data['categories']의 타입:", type(data['categories']))
    print("data['categories']의 내용:", data['categories'])
else:
    print("'categories' 키가 존재하지 않습니다.")

# 전체 데이터를 너무 길게 출력하지 않도록 일부만 확인
print("JSON 데이터 일부:", json.dumps(data, indent=2)[:500])  # 500자만 출력
