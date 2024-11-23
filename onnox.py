import cv2
import numpy as np
import onnxruntime as ort

# 모델 경로 및 이미지 경로
model_path = "/Users/lch/development/opencv/finalProject/yolov5/runs/train/exp28/weights/best.onnx"
image_path = "apple_c.jpg"

# ONNX 모델 로드
session = ort.InferenceSession(model_path)
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

# 이미지 로드 및 전처리
image = cv2.imread(image_path)
image_height, image_width = image.shape[:2]
blob = cv2.dnn.blobFromImage(
    image, scalefactor=1/255.0, size=(640, 640), swapRB=True, crop=False
)

# 모델 추론
outputs = session.run([output_name], {input_name: blob})[0]  # (1, 25200, 10)
detections = outputs[0]  # 배치 차원 제거 (25200, 10)

# 탐지 결과 처리
confidence_threshold = 0.25
detected_objects = []

for detection in detections:
    x_center, y_center, w, h, confidence, *class_scores = detection

    # 신뢰도 필터링
    if confidence > confidence_threshold:
        class_id = int(np.argmax(class_scores))  # 클래스 ID 추출

        # 정규화된 좌표를 원본 이미지 크기로 변환
        x1 = int((x_center - w / 2) * image_width)
        y1 = int((y_center - h / 2) * image_height)
        x2 = int((x_center + w / 2) * image_width)
        y2 = int((y_center + h / 2) * image_height)

        # 유효한 탐지만 저장
        if 0 <= x1 < image_width and 0 <= y1 < image_height:
            detected_objects.append((class_id, confidence, x1, y1, x2, y2))

# 탐지 결과 시각화
for obj in detected_objects:
    class_id, confidence, x1, y1, x2, y2 = obj
    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
    label = f"Class {class_id}: {confidence:.2f}"
    cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

# 결과 저장 및 출력
output_image_path = "./detected_notes_visualized.jpg"
cv2.imwrite(output_image_path, image)
print(f"Detection visualization saved to {output_image_path}")

cv2.imshow("Detection Visualization", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
