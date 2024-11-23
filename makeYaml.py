with open('/Users/lch/development/opencv/finalProject/data.yaml', 'w') as f:
    f.write("train: /Users/lch/development/opencv/finalProject/yolo_dataset/train\n")
    f.write("val: /Users/lch/development/opencv/finalProject/yolo_dataset/val\n\n")
    f.write("nc: 185\n")
    f.write("names:\n")
    for i in range(185):
        f.write(f"  - class{i}\n")
print("data.yaml 작성 완료!")
