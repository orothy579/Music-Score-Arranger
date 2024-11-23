from albumentations import (
    Compose, RandomBrightnessContrast, HueSaturationValue, ShiftScaleRotate,
    HorizontalFlip, GridDistortion, BboxParams
)

import cv2


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

# Albumentations 변환 정의 (바운딩 박스 포함)
transform = Compose(
    [
        RandomBrightnessContrast(p=0.5),
        HueSaturationValue(hue_shift_limit=15, sat_shift_limit=30, val_shift_limit=20, p=0.5),
        ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, p=0.7),
        HorizontalFlip(p=0.5),
        GridDistortion(p=0.3),
    ],
    bbox_params=BboxParams(format='yolo', label_fields=['class_labels'])
)

# 원본 이미지 읽기
image = cv2.imread('/Users/lch/development/opencv/finalProject/apple_dataset/train/images/apple_c.jpg')
image_height, image_width = image.shape[:2]

# 증강된 이미지와 라벨 생성
for i in range(50):  # 10개의 증강된 이미지 생성
    augmented = transform(
        image=image,
        bboxes=[label[1:] for label in original_labels],  # 좌표만 전달
        class_labels=[label[0] for label in original_labels],  # 클래스 ID만 전달
    )
    augmented_image = augmented['image']
    augmented_bboxes = augmented['bboxes']
    augmented_classes = augmented['class_labels']

    # 증강된 이미지 저장
    output_image_path = f'/Users/lch/development/opencv/finalProject/apple_dataset/train/images/augmented_image_{i}.jpg'
    cv2.imwrite(output_image_path, augmented_image)

    # 증강된 라벨 저장 (YOLO 포맷)
    output_label_path = f'/Users/lch/development/opencv/finalProject/apple_dataset/train/labels/augmented_image_{i}.txt'
    with open(output_label_path, 'w') as f:
        for bbox, class_id in zip(augmented_bboxes, augmented_classes):
            bbox_str = ' '.join(map(str, bbox))  # 바운딩 박스 좌표를 문자열로 변환
            f.write(f"{class_id} {bbox_str}\n")

print("증강된 이미지와 라벨 파일 생성 완료!")
