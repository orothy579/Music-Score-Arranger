import torch
import cv2
import numpy as np

# 음표 이미지 로드
notes_path = "notes/"
eighth_note = cv2.imread(notes_path + "eighthNote.png", cv2.IMREAD_UNCHANGED)
eighth_note_ti = cv2.imread(notes_path + "eighthNoteTi.png", cv2.IMREAD_UNCHANGED)
quarter_note = cv2.imread(notes_path + "quarterNote.png", cv2.IMREAD_UNCHANGED)
quarter_note_ti = cv2.imread(notes_path + "quarterNoteTi.png", cv2.IMREAD_UNCHANGED)
treble_clef = cv2.imread(notes_path + "trebleClef.png", cv2.IMREAD_UNCHANGED)

# PyTorch 모델 로드
model_path = "/Users/lch/development/opencv/finalProject/yolov5/runs/train/exp28/weights/best.pt"
model = torch.hub.load('./yolov5', 'custom', path=model_path, source='local')

# 이미지 로드 및 전처리
image_path = "apple_c.jpg"
image = cv2.imread(image_path)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# PyTorch 추론
results = model(image_rgb)
detections = results.xyxy[0].numpy()  # 탐지된 객체 좌표

# 새 이미지 생성
new_image = np.ones_like(image) * 255  # 흰색 배경 이미지

# 오선 간격 계산 변수
staff_spacing = None
staff_y1 = None


def is_ti(center_y, staff_y1, staff_spacing):
    """
    음표가 뒤집혀야 하는지 확인 (Ti 여부).
    """
    staff_center_y = staff_y1 + staff_spacing * 2.5
    return center_y <= staff_center_y


def overlay_image(background, overlay, x, y, width, height):
    """
    background에 overlay 이미지를 합성.
    """
    overlay_resized = cv2.resize(overlay, (width, height), interpolation=cv2.INTER_AREA)
    for i in range(overlay_resized.shape[0]):  # height
        for j in range(overlay_resized.shape[1]):  # width
            if overlay_resized[i, j][3] > 0:  # 투명하지 않은 픽셀
                background[y + i, x + j] = overlay_resized[i, j][:3]


def draw_staff(image, bounding_box):
    """
    탐지된 오선 박스를 기반으로 오선을 그립니다.
    """
    global staff_spacing, staff_y1
    x1, y1, x2, y2 = bounding_box
    staff_y1 = y1
    staff_spacing = (y2 - y1) / 4  # 오선 간격 계산
    for i in range(5):  # 오선 5줄 그림
        y = int(y1 + i * staff_spacing)
        cv2.line(image, (int(x1), y), (int(x2), y), (0, 0, 0), 2)


def draw_treble_clef_image(image, bounding_box):
    """
    높은 음자리표를 추가합니다.
    """
    x1, y1, x2, y2 = bounding_box
    width = int(x2 - x1)
    height = int(y2 - y1)
    overlay_image(image, treble_clef, int(x1), int(y1), width, height)


def draw_note_image(image, bounding_box, note_type, is_ti):
    """
    음표 이미지를 추가합니다.
    """
    x1, y1, x2, y2 = bounding_box
    width = int(x2 - x1)
    height = int(y2 - y1)
    if note_type == "eighth":
        overlay = eighth_note_ti if is_ti else eighth_note
    elif note_type == "quarter":
        overlay = quarter_note_ti if is_ti else quarter_note
    else:
        return
    overlay_image(image, overlay, int(x1), int(y1), width, height)


# 탐지 결과 처리
for detection in detections:
    x1, y1, x2, y2, confidence, class_id = detection
    if confidence > 0.33:
        # 기존 Detection 이미지 출력
        label = f"Class {int(class_id)}: {confidence:.2f}"
        cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
        cv2.putText(image, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        # 새로운 이미지에 추가
        if class_id == 0:  # 오선
            draw_staff(new_image, (x1, y1, x2, y2))
        elif class_id == 1:  # 높은 음자리표
            draw_treble_clef_image(new_image, (x1, y1, x2, y2))
        elif class_id == 2:  # 8분 음표
            center_y = (y1 + y2) // 2
            draw_note_image(new_image, (x1, y1, x2, y2), "eighth", is_ti(center_y, staff_y1, staff_spacing))
        elif class_id == 3:  # 4분 음표
            center_y = (y1 + y2) // 2
            draw_note_image(new_image, (x1, y1, x2, y2), "quarter", is_ti(center_y, staff_y1, staff_spacing))

# 기존 Detection 이미지 출력
cv2.imshow("Detections", image)
cv2.imwrite("detected_image.jpg", image)

# 새 이미지 출력
cv2.imshow("New Image", new_image)
cv2.imwrite("new_image.jpg", new_image)

cv2.waitKey(0)
cv2.destroyAllWindows()
