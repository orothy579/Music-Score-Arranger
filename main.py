import torch
import cv2
import numpy as np

# 음표 이미지 로드
notes_path = "notes/"
eighth_note = cv2.imread(notes_path + "eighthNote.png", cv2.IMREAD_UNCHANGED)
eighth_note_ti = cv2.imread(
    notes_path + "eighthNoteTi.png", cv2.IMREAD_UNCHANGED)
quarter_note = cv2.imread(notes_path + "quarterNote.png", cv2.IMREAD_UNCHANGED)
quarter_note_ti = cv2.imread(
    notes_path + "quarterNoteTi.png", cv2.IMREAD_UNCHANGED)
half_note = cv2.imread(notes_path + "halfNote.png",  cv2.IMREAD_UNCHANGED)
treble_clef = cv2.imread(notes_path + "trebleClef.png", cv2.IMREAD_UNCHANGED)

# PyTorch 모델 로드
model_path = "/Users/lch/development/opencv/finalProject/yolov5/runs/train/exp29/weights/best.pt"
model = torch.hub.load('./yolov5', 'custom', path=model_path, source='local')

# 이미지 로드 및 전처리
image_path = "apple_c.jpg"
image = cv2.imread(image_path)

# 대비 조정 및 전처리
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # 흑백 변환
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
enhanced = clahe.apply(gray)  # CLAHE 적용

# 노이즈 제거
blurred = cv2.GaussianBlur(enhanced, (5, 5), 0)


# 전처리된 이미지로 YOLO 모델 실행
image_rgb = cv2.cvtColor(blurred, cv2.COLOR_GRAY2RGB)  # YOLO는 RGB 이미지를 입력으로 사용
results = model(image_rgb)

# PyTorch 추론
results = model(image_rgb)
detections = results.xyxy[0].numpy()  # 탐지된 객체 좌표

# 새 이미지 생성
new_image = np.ones_like(image) * 255  # 흰색 배경 이미지

# 오선 간격 계산 변수
staff_spacing = 10
staff_y1 = 10


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
    overlay_resized = cv2.resize(
        overlay, (width, height), interpolation=cv2.INTER_AREA)
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

    # 이미지의 높이와 너비 가져오기
    image_height, image_width = image.shape[:2]

    # None 처리: 값이 없으면 기본값 할당
    if x1 is None:
        x1 = 0
    if x2 is None:
        x2 = image_width
    if y1 is None:
        y1 = 0
    if y2 is None:
        y2 = image_height - 10  # 이미지 높이 - 10

    # x와 y 값의 범위 제한
    x1 = max(0, min(x1, image_width))
    x2 = max(0, min(x2, image_width))
    y1 = max(0, min(y1, image_height))
    y2 = max(0, min(y2, image_height))

    # staff_y1과 staff_spacing 계산
    staff_y1 = y1
    staff_spacing = (y2 - y1) / 4 if (y2 -
                                      y1) > 0 else image_height / 20  # 기본 간격

    print(f"staff_y1: {staff_y1}, staff_spacing: {staff_spacing}")

    # 오선 그리기
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
    elif note_type == "half":
        overlay = half_note
    else:
        return
    overlay_image(image, overlay, int(x1), int(y1), width, height)


# 탐지 결과를 x1 기준으로 정렬
sorted_detections = sorted(detections, key=lambda det: det[0])  # x1을 기준으로 정렬

# 탐지 결과 처리
for detection in sorted_detections:
    x1, y1, x2, y2, confidence, class_id = detection

    if confidence > 0.35:
        # center_y 계산
        center_y = (y1 + y2) // 2

        # center_y 출력
        print(f"Class {int(class_id)} | x1: {int(x1)} | center_y: {int(center_y)}")

        # 기존 Detection 이미지 출력
        label = f"Class {int(class_id)}"
        cv2.rectangle(image, (int(x1), int(y1)),
                      (int(x2), int(y2)), (0, 0, 255), 2)
        cv2.putText(image, label, (int(x1), int(y1) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        # 새로운 이미지에 추가
        if class_id == 0:  # 오선
            draw_staff(new_image, (x1, y1, x2, y2))
        elif class_id == 1:  # 높은 음자리표
            draw_treble_clef_image(new_image, (x1, y1, x2, y2))
        elif class_id == 2:  # 8분 음표
            draw_note_image(new_image, (x1, y1, x2, y2), "eighth",
                            is_ti(center_y, staff_y1, staff_spacing))
        elif class_id == 3:  # 4분 음표
            draw_note_image(new_image, (x1, y1, x2, y2), "quarter",
                            is_ti(center_y, staff_y1, staff_spacing))
        elif class_id == 4:  # 2분 음표
            draw_note_image(new_image, (x1, y1, x2, y2), "half",
                            is_ti(center_y, staff_y1, staff_spacing))

# 기존 Detection 이미지 출력
cv2.imshow("Detections", image)
cv2.imwrite("Images/detected_image.jpg", image)

# 새 이미지 (new_image)에 텍스트와 도형 추가
from PIL import Image, ImageDraw, ImageFont

# OpenCV 이미지 (BGR)를 PIL 이미지 (RGB)로 변환
pil_image = Image.fromarray(cv2.cvtColor(new_image, cv2.COLOR_BGR2RGB))
draw = ImageDraw.Draw(pil_image)

# 폰트 설정
try:
    font_path = "/System/Library/Fonts/Supplemental/Helvetica.ttc"
    font = ImageFont.truetype(font_path, 36)  # 폰트 크기 36
    large_font = ImageFont.truetype(font_path, 60)  # 박자용 폰트 크기를 54로 설정

except:
    print("기본 폰트를 사용합니다. (크기 조정 불가)")
    font = ImageFont.load_default()

# 흰색 직사각형 좌표 설정
rect_top_left = (180, 180)  # 흰색 직사각형의 왼쪽 상단 좌표
rect_bottom_right = (1600, 300)  # 흰색 직사각형의 오른쪽 하단 좌표

# 흰색 직사각형 그리기
draw.rectangle([rect_top_left, rect_bottom_right], fill="white")

draw.text((170, 310), "3", fill="black", font=large_font)  # 박자 추가
draw.text((170, 350), "4", fill="black", font=large_font)  # 박자 추가

# 텍스트 추가
draw.text((230, 260), "F", fill="black", font=font)
draw.text((550, 260), "C", fill="black", font=font)
draw.text((900, 260), "C7", fill="black", font=font)
draw.text((1300, 260), "F", fill="black", font=font)

# PIL 이미지를 OpenCV 이미지로 다시 변환 (BGR로 변환)
new_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

# 마디선의 x 좌표 설정
bar_positions = [530, 870, 1270]  # F, C, C7, F 사이의 마디선 x 좌표

# 마디선의 y 좌표 설정
bar_y_start = 305  # 수직선 시작 y 좌표
bar_y_end = 405    # 수직선 끝 y 좌표

# 이미지에 마디선 그리기
line_thickness = 2  # 선 두께
for x in bar_positions:
    cv2.line(new_image, (x, bar_y_start), (x, bar_y_end), (0, 0, 0), thickness=line_thickness)  # y 좌표 적용

# 수정된 new_image 저장 및 같은 창에 표시
cv2.imwrite("Images/new_image.jpg", new_image)
cv2.imshow("New Image", new_image)  # 기존 창 이름 사용

cv2.waitKey(0)
cv2.destroyAllWindows()
