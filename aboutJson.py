import json

# JSON 파일 경로
json_file = "./ds2_dense/deepscores_train.json"

# JSON 파일 읽기
with open(json_file, 'r') as f:
    data = json.load(f)

# JSON 데이터의 상위 키 확인
print(data.keys())

# # JSON 파일의 'images' 항목의 첫 번째 데이터 확인
# print(data['images'][0])


for ann in data['annotations']:
    print(type(ann))  # 각 항목의 타입 확인
    print(ann)        # 첫 번째 항목 내용 출력
    break

# print(type(data['annotations']))
# print(data['annotations'])

# annotations의 첫 번째 항목 확인
# print(data['annotations'][0])

print(type(data['annotations']))  # 딕셔너리인지 확인
# print(data['annotations'].keys())  # 딕셔너리의 키 확인

