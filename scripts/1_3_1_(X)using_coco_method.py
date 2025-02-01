import os
import numpy as np

# 설정
LABELS_DIR = "/home/a/A_2024_selfcode/PCB_yolo/dataset/4_labels"
OUTPUT_LABELS_DIR = "/home/a/A_2024_selfcode/PCB_yolo/dataset/5_coco_filtered_labels"

ORIGINAL_IMAGE_WIDTH = 3904
ORIGINAL_IMAGE_HEIGHT = 3904

SMALL_THRESHOLD = 32**2
MEDIUM_THRESHOLD = 96**2

os.makedirs(OUTPUT_LABELS_DIR, exist_ok=True)

# 1. 폴리곤 면적 계산 함수
def polygon_area(x_coords, y_coords):
    n = len(x_coords)
    return 0.5 * abs(
        sum(x_coords[i] * y_coords[(i + 1) % n] - x_coords[(i + 1) % n] * y_coords[i] 
        for i in range(n))
    )

# 2. 작은 및 중간 객체 필터링
for file_name in os.listdir(LABELS_DIR):
    input_path = os.path.join(LABELS_DIR, file_name)
    output_path = os.path.join(OUTPUT_LABELS_DIR, file_name)

    with open(input_path, 'r') as file, open(output_path, 'w') as output_file:
        for line in file.readlines():
            data = line.strip().split()  # 양 끝의 공백 제거
            if not data:  # 빈 줄 건너뛰기
                continue
            class_id = int(data[0])
            points = list(map(float, data[1:]))

            # OBB 좌표 추출 (원본 이미지 픽셀 단위)
            x_coords = [x * ORIGINAL_IMAGE_WIDTH for x in points[0::2]]
            y_coords = [y * ORIGINAL_IMAGE_HEIGHT for y in points[1::2]]

            # 폴리곤 면적 계산
            area_px = polygon_area(x_coords, y_coords)

            # 면적 비교 (픽셀 기준)
            if SMALL_THRESHOLD <= area_px <= MEDIUM_THRESHOLD:
                output_file.write(line.strip() + '\n')  # 기존 줄바꿈 제거 후 새로 추가

    print(f"Processed {input_path} → {output_path} (Small and Medium Objects Only)")

print("✅ 라벨 정제 완료: 원본 이미지 기준 작은 및 중간 객체만 포함된 데이터셋 생성")
