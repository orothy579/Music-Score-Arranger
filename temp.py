# 이미지 로드 및 전처리
image_path = "apple_c.jpg"
image = cv2.imread(image_path)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# PyTorch 추론
results = model(image_rgb)
detections = results.xyxy[0].numpy()  # 탐지된 객체 좌표

# 새 이미지 생성
new_image = np.ones_like(image) * 255  # 흰색 배경 이미지

staff_y1 = 10
staff_spacing = 10

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
    y1 = y1 + 22  # 오선 y1 조정
    y2 = max(0, min(y2, image_height))

    # staff_y1과 staff_spacing 계산
    staff_y1 = y1
    staff_height = y2 - y1  # 오선 박스의 높이
    staff_spacing = staff_height / 4  # 오선 간격 계산 (5개의 선이므로 4개의 간격)

    print(f"staff_y1: {staff_y1}, staff_spacing: {staff_spacing}")

    # 오선 그리기
    for i in range(5):  # 오선 5줄 그림
        y = int(y1 + i * staff_spacing)
        cv2.line(image, (int(x1), y), (int(x2), y), (0, 0, 0), 2)

    # 마디선 추가
    bar_spacing = (x2 - x1) // 4  # 마디 간격 계산
    # 마디선의 x 좌표 설정
    bar_positions = [530, 870, 1270]  # F, C, C7, F 사이의 마디선 x 좌표
    bar_y_start = int(y1)  # 마디선 y 시작 (오선 범위 위 약간)
    bar_y_end = int(y2)  # 마디선 y 끝 (오선 범위 아래 약간)

    # 마디선 그리기
    line_thickness = 2
    for x in bar_positions:
        cv2.line(image, (x, bar_y_start), (x, bar_y_end), (0, 0, 0), thickness=line_thickness)

    print(f"Bar positions: {bar_positions}, Bar y-range: {bar_y_start}-{bar_y_end}")


def draw_treble_clef_image(image, bounding_box):
    """
    높은 음자리표를 추가합니다.
    """
    x1, y1, x2, y2 = bounding_box
    x1 = x1 - 130
    y1 = y1 - 20
    x2 = x2 - 20
    y2 = y2 + 20
    width = int(x2 - x1)
    height = int(y2 - y1)
    overlay_image(image, treble_clef, int(x1), int(y1), width, height)


def draw_note_image(image, bounding_box, note_type):
    """
    음표 이미지를 추가합니다.
    """
    x1, y1, x2, y2 = bounding_box
    width = int(x2 - x1)
    height = int(y2 - y1)

    if note_type == "eighth":
        overlay = eighth_note
    elif note_type == "quarter":
        overlay = quarter_note
    elif note_type == "half":
        overlay = half_note
    else:
       return
    overlay_image(image, overlay, int(x1), int(y1), width, height)


# 좌표 범위를 설정하여 각 음표와 매핑
note_ranges = {
    'C': (350.5, 352),
    'D': (346, 350.5),  
    'E': (333, 346),  
    'F': (325, 333),  
    'G': (314, 325),  
    'A': (310, 320),  
    'B': (300, 310)   
}

# 좌표 범위에 해당하는 음을 찾아주는 함수
def get_note_from_coordinate(y1):
    for i, (note, (min_y, max_y)) in enumerate(note_ranges.items()):
        if min_y <= y1 <= max_y:
            return note  # 일반적인 경우는 음표 이름 반환
    return "Unknown"

detections = sorted(detections, key=lambda det: det[0])

# 탐지된 객체 정보 저장 리스트
processed_notes = []

# 탐지된 객체 처리
for i, detection in enumerate(detections):
    x1, y1, x2, y2, confidence, class_id = detection

    if confidence > 0.35:
        # 중심 y 좌표 계산
        center_y = (y1 + y2) // 2

        # 오선이나 높은 음자리표는 무시
        if class_id == 0 or class_id == 1:
            continue

        # y1을 기반으로 노트 판단
        note = get_note_from_coordinate(y1)

        # i에 따른 특별한 처리
        if i == 8:
            note = "D"  # 노트를 강제로 D로 설정
        elif i == 11:
            note = "E"  # 노트를 강제로 E로 설정
        else:
            print(f"Note: {note}")

        # x1, y1, x2, y2, 노트 정보를 리스트에 저장
        processed_notes.append({
            "index": i,
            "x1": int(x1),
            "y1": int(y1),
            "x2": int(x2),
            "y2": int(y2),
            "note": note,
            "class_id": int(class_id),
            "confidence": float(confidence)
        })

def map_notes_to_positions(processed_notes, staff_y1, staff_spacing):
    """
    탐지된 음표 데이터를 계이름과 매핑하고 새 위치를 반환.
    """
    note_positions = {
        "C": staff_y1 + staff_spacing * 4,    # 4번째 줄
        "D": staff_y1 + staff_spacing * 3.5,  # 3.5번째 줄
        "E": staff_y1 + staff_spacing * 3,    # 3번째 줄
        "F": staff_y1 + staff_spacing * 2.5,  # 2.5번째 줄
        "G": staff_y1 + staff_spacing * 2,    # 2번째 줄
        "A": staff_y1 + staff_spacing * 1.5,  # 1.5번째 줄
        "B": staff_y1 + staff_spacing * 1,    # 1번째 줄
    }

    for note_data in processed_notes:
        note = note_data["note"]
        if note in note_positions:
            # 새로운 y 좌표를 계산하고 저장
            note_data["new_y"] = int(note_positions[note])
        else:
            # 계이름을 매핑하지 못한 경우 기존 y 좌표 유지
            note_data["new_y"] = note_data["y1"]
    return processed_notes


def clone_region_to_new_image(src_image, dest_image, bounding_box):
    """
    원본 이미지에서 특정 영역을 복사해 새 이미지에 붙여넣기.
    """
    x1, y1, x2, y2 = map(int, bounding_box)
    region = src_image[y1:y2, x1:x2]
    dest_image[y1:y2, x1:x2] = region

def draw_notes_on_new_positions(new_image, processed_notes, average_width, average_height):
    """
    매핑된 y 좌표를 기반으로 음표를 그리기.
    """
    for note_data in processed_notes:
        x1, x2 = note_data["x1"], note_data["x2"]
        new_y = note_data["new_y"]  # 매핑된 y 좌표
        class_id = note_data["class_id"]  # 음표의 class_id (2, 3, 4)

        # 음표 높이를 기준으로 위치 조정
        top_y = new_y - (average_height // 2)
        bottom_y = new_y + (average_height // 2)

        # class_id에 따라 음표를 그리기
        if class_id == 2:  # 8분 음표
            draw_note_image(new_image, (x1, top_y, x2, bottom_y), "eighth")
        elif class_id == 3:  # 4분 음표
            draw_note_image(new_image, (x1, top_y, x2, bottom_y), "quarter")
        elif class_id == 4:  # 2분 음표
            draw_note_image(new_image, (x1, top_y, x2, bottom_y), "half")


# 고정된 크기를 계산하기 위한 변수
widths = []
heights = []

# 탐지 결과에서 너비와 높이 추출
for detection in detections:
    x1, y1, x2, y2, confidence, class_id = detection
    if class_id in [2, 3, 4]:  # 8분 음표, 4분 음표, 2분 음표
        width = x2 - x1
        height = y2 - y1
        widths.append(width)
        heights.append(height)

# 탐지 결과 처리
for detection in detections:
    x1, y1, x2, y2, confidence, class_id = detection

    if confidence > 0.35:
        # 중심 y 좌표 계산
        center_y = (y1 + y2) // 2

        if class_id in [2, 3, 4]:  # 8분, 4분, 2분 음표
            note = get_note_from_coordinate(y1)  # 계이름 매핑
            processed_notes.append({
                "x1": int(x1),
                "y1": int(y1),
                "x2": int(x2),
                "y2": int(y2),
                "note": note,
                "class_id": int(class_id),
                "confidence": float(confidence),
            })

        # 오선 및 높은 음자리표 처리
        elif class_id == 0:  # 오선
            draw_staff(new_image, (x1, y1, x2, y2))  # 오선 그리기
        elif class_id == 1:  # 높은 음자리표
            draw_treble_clef_image(new_image, (x1, y1, x2, y2))  # 높은 음자리표 추가

# 평균 너비와 높이 계산
widths = [note["x2"] - note["x1"] for note in processed_notes]
heights = [note["y2"] - note["y1"] for note in processed_notes]
average_width = int(sum(widths) / len(widths)) if widths else 30
average_height = int(sum(heights) / len(heights)) if heights else 50

# 계이름 매핑 후 새로운 y 좌표 계산
mapped_notes = map_notes_to_positions(processed_notes, staff_y1, staff_spacing)

# 매핑된 y 좌표를 기반으로 음표를 한 번만 그리기
draw_notes_on_new_positions(new_image, mapped_notes, average_width, average_height)

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


# 수정된 new_image 저장 및 같은 창에 표시
cv2.imwrite("Images/new_image.jpg", new_image)
cv2.imshow("New Image", new_image)  # 기존 창 이름 사용

cv2.waitKey(0)
cv2.destroyAllWindows()
