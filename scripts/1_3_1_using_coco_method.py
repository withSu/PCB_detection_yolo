import os
import numpy as np

# 설정
LABELS_DIR = "/home/a/A_2024_selfcode/PCB_yolo/dataset/4_labels"  # 원본 라벨 파일 경로
OUTPUT_LABELS_DIR = "/home/a/A_2024_selfcode/PCB_yolo/dataset/5_coco_filtered_labels"  # 정제된 라벨 파일 저장 경로

ORIGINAL_IMAGE_WIDTH = 3904  # 원본 이미지 가로 크기
ORIGINAL_IMAGE_HEIGHT = 3904  # 원본 이미지 세로 크기
RESIZED_SHORT_EDGE = 800  # 리사이즈 후 짧은 변 크기 (DETR 방식)

# COCO 기준 설정 (pixel², 리사이즈된 크기 기준)
SMALL_THRESHOLD = 32**2
MEDIUM_THRESHOLD = 96**2

# 출력 디렉토리 생성
os.makedirs(OUTPUT_LABELS_DIR, exist_ok=True)

# 1. 리사이즈 비율 계산
resize_ratio = RESIZED_SHORT_EDGE / ORIGINAL_IMAGE_WIDTH  # 가로를 기준으로 계산
resized_width = RESIZED_SHORT_EDGE
resized_height = int(ORIGINAL_IMAGE_HEIGHT * resize_ratio)

# 2. 작은 및 중간 객체 필터링
for file_name in os.listdir(LABELS_DIR):
    input_path = os.path.join(LABELS_DIR, file_name)
    output_path = os.path.join(OUTPUT_LABELS_DIR, file_name)

    with open(input_path, 'r') as file, open(output_path, 'w') as output_file:
        for line in file.readlines():
            data = line.split()
            class_id = int(data[0])
            points = list(map(float, data[1:]))

            # OBB 영역 계산 (리사이즈된 크기 기준)
            x_coords = [x * resized_width for x in points[0::2]]  # x좌표 리사이즈
            y_coords = [y * resized_height for y in points[1::2]]  # y좌표 리사이즈
            width_px = max(x_coords) - min(x_coords)
            height_px = max(y_coords) - min(y_coords)
            area_px = width_px * height_px

            # COCO 기준에 따라 작은/중간 객체만 기록
            if SMALL_THRESHOLD <= area_px <= MEDIUM_THRESHOLD:
                output_file.write(line)

    print(f"Processed {input_path} → {output_path} (Small and Medium Objects Only)")

print("✅ 라벨 정제 완료: 작은 및 중간 객체만 포함된 데이터셋 생성")
