from albumentations import (
    Compose, RandomBrightnessContrast, HueSaturationValue, ShiftScaleRotate,
    HorizontalFlip, GridDistortion, GaussianBlur, MotionBlur, BboxParams, CoarseDropout, InvertImg
)
import cv2
import os

# 원래 라벨 데이터를 YOLO 포맷으로 정의
original_labels = [
    [0, 0.49765625, 0.5671875, 0.92421875, 0.153125],
    [1, 0.0703125, 0.5703125, 0.0296875, 0.22890625],
    [2, 0.13828125, 0.6125, 0.03515625, 0.1328125],
    [2, 0.18125, 0.60234375, 0.028125, 0.13125],
    [3, 0.21796875, 0.58359375, 0.0234375, 0.13046875],
    [3, 0.26484375, 0.5796875, 0.025, 0.121875],
    [2, 0.3453125, 0.57421875, 0.03125, 0.13203125],
    [2, 0.3859375, 0.58046875, 0.028125, 0.12421875],
    [4, 0.42421875, 0.59765625, 0.02109375, 0.13359375],
    [2, 0.55859375, 0.6015625, 0.0296875, 0.13046875],
    [2, 0.6, 0.584375, 0.02890625, 0.14328125],
    [3, 0.6375, 0.57578125, 0.02421875, 0.1546875],
    [3, 0.69296875, 0.57265625, 0.0296875, 0.14328125],
    [2, 0.78359375, 0.55390625, 0.02890625, 0.15078125],
    [2, 0.8203125, 0.5671875, 0.03125, 0.1421875],
    [4, 0.85859375, 0.58359375, 0.02265625, 0.13984375]
]

# 데이터 증강 정의
transform = Compose(
    [
        RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.7),  # 밝기와 대비 조정
        HueSaturationValue(hue_shift_limit=20, sat_shift_limit=50, val_shift_limit=30, p=0.6),  # 색상 변환
        ShiftScaleRotate(shift_limit=0.2, scale_limit=0.2, rotate_limit=45, border_mode=cv2.BORDER_CONSTANT, p=0.8),
        HorizontalFlip(p=0.5),  # 좌우 반전
        GridDistortion(num_steps=5, distort_limit=0.4, p=0.4),  # 격자 왜곡
        GaussianBlur(blur_limit=(3, 5), p=0.5),  # 가우시안 블러
        MotionBlur(blur_limit=(3, 7), p=0.4),  # 모션 블러
        CoarseDropout(max_holes=5, max_height=15, max_width=15, p=0.3),  # 랜덤 노이즈
        InvertImg(p=0.3),  # 이미지 네거티브
    ],
    bbox_params=BboxParams(format='yolo', label_fields=['class_labels'])
)

# 이미지 읽기
input_image_path = '/Users/lch/development/opencv/finalProject/apple_dataset/train/images/apple_c.jpg'
image = cv2.imread(input_image_path)
image_height, image_width = image.shape[:2]

# 증강된 이미지와 라벨 생성
output_image_dir = '/Users/lch/development/opencv/finalProject/apple_dataset/train/images/'
output_label_dir = '/Users/lch/development/opencv/finalProject/apple_dataset/train/labels/'
os.makedirs(output_image_dir, exist_ok=True)
os.makedirs(output_label_dir, exist_ok=True)

for i in range(100):  # 100개의 증강된 이미지 생성
    augmented = transform(
        image=image,
        bboxes=[label[1:] for label in original_labels],  # 바운딩 박스 좌표만 전달
        class_labels=[label[0] for label in original_labels],  # 클래스 ID만 전달
    )
    augmented_image = augmented['image']
    augmented_bboxes = augmented['bboxes']
    augmented_classes = augmented['class_labels']

    # 증강된 이미지 저장
    output_image_path = os.path.join(output_image_dir, f'augmented_image_{i}.jpg')
    cv2.imwrite(output_image_path, augmented_image)

    # 증강된 라벨 저장 (YOLO 포맷)
    output_label_path = os.path.join(output_label_dir, f'augmented_image_{i}.txt')
    with open(output_label_path, 'w') as f:
        for bbox, class_id in zip(augmented_bboxes, augmented_classes):
            bbox_str = ' '.join(map(str, bbox))  # 바운딩 박스 좌표를 문자열로 변환
            f.write(f"{class_id} {bbox_str}\n")

print("증강된 이미지와 라벨 파일 생성 완료!")
