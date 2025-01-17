import os
import numpy as np
from skimage.filters import threshold_otsu
import matplotlib.pyplot as plt

# 설정
LABELS_DIR = "/home/a/A_2024_selfcode/PCB/dataset/4_labels"  # 원본 라벨 파일 경로
OUTPUT_LABELS_DIR = "/home/a/A_2024_selfcode/PCB/dataset/5_after_otsu_only_under_threshold_labels"  # 정제된 라벨 파일 저장 경로

IMAGE_WIDTH = 3904  # 이미지 가로 크기
IMAGE_HEIGHT = 3904  # 이미지 세로 크기

# 출력 디렉토리 생성
os.makedirs(OUTPUT_LABELS_DIR, exist_ok=True)

# 1. Bounding Box 데이터 추출
bounding_box_data = []

for file_name in os.listdir(LABELS_DIR):
    file_path = os.path.join(LABELS_DIR, file_name)
    with open(file_path, 'r') as file:
        for line in file.readlines():
            data = line.split()
            class_id = int(data[0])  # 클래스 ID
            points = list(map(float, data[1:]))  # 좌표 데이터 (x1 y1 x2 y2 x3 y3 x4 y4)
            
            # OBB 영역 계산
            x_coords = points[0::2]
            y_coords = points[1::2]
            width_px = max(x_coords) - min(x_coords)
            height_px = max(y_coords) - min(y_coords)
            area = width_px * height_px  # OBB의 Bounding Box 면적
            bounding_box_data.append({"file": file_name, "class_id": class_id, "area": area, "line": line})

# 2. Otsu's Method로 임계값 계산
areas_normalized = [entry["area"] for entry in bounding_box_data]  # 정규화된 면적 리스트
threshold_normalized = threshold_otsu(np.array(areas_normalized))
print(f"Otsu's Threshold (Normalized): {threshold_normalized:.6f} (normalized)")

# 실제 픽셀 단위 면적 계산
areas_in_pixels = []  # 실제 픽셀 단위 면적 리스트
for entry in bounding_box_data:
    points = list(map(float, entry["line"].split()[1:]))  # 라벨 데이터에서 좌표 추출
    x_coords = [x * IMAGE_WIDTH for x in points[0::2]]  # x좌표 복원
    y_coords = [y * IMAGE_HEIGHT for y in points[1::2]]  # y좌표 복원
    width_px = max(x_coords) - min(x_coords)
    height_px = max(y_coords) - min(y_coords)
    area_px = width_px * height_px  # 실제 픽셀 단위 면적
    areas_in_pixels.append(area_px)

threshold_pixels = threshold_otsu(np.array(areas_in_pixels))
print(f"Otsu's Threshold (Pixels): {threshold_pixels:.2f} pixels²")

# 2-1. 정규화된 히스토그램 생성
plt.figure(figsize=(10, 6))
plt.hist(areas_normalized, bins=50, color='blue', alpha=0.7, label="Bounding Box Areas (Normalized)")
plt.axvline(x=threshold_normalized, color='red', linestyle='--', label=f"Otsu's Threshold ({threshold_normalized:.6f})")
plt.title("Histogram of Bounding Box Areas (Normalized, Log Scale)")
plt.xlabel("Area (normalized)")
plt.ylabel("Frequency (log scale)")
plt.yscale("log")  # y축 로그 스케일 적용
plt.legend()
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.show()

# 2-2. 픽셀 단위 히스토그램 생성
plt.figure(figsize=(10, 6))
plt.hist(
    areas_in_pixels,
    bins=50,
    color='green',
    alpha=0.7,
    label="Bounding Box Areas (Pixels)"
)
plt.axvline(
    x=threshold_pixels,
    color='red',
    linestyle='--',
    label=f"Otsu's Threshold ({threshold_pixels:.2f} px²)"
)
plt.title("Histogram of Bounding Box Areas (Pixels, Log Scale)")
plt.xlabel("Area (pixels²)")
plt.ylabel("Frequency (log scale)")
plt.yscale("log")  # y축 로그 스케일 적용
plt.xlim(0, max(areas_in_pixels))  # x축 범위를 실제 최대 값으로 설정
plt.ticklabel_format(style='plain', axis='x')  # x축 레이블을 일반 숫자로 표시
plt.legend()
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.show()


# 3. 작은 소자만 포함된 라벨 파일 생성
for file_name in os.listdir(LABELS_DIR):
    input_path = os.path.join(LABELS_DIR, file_name)
    output_path = os.path.join(OUTPUT_LABELS_DIR, file_name)
    
    with open(input_path, 'r') as file, open(output_path, 'w') as output_file:
        for line in file.readlines():
            data = line.split()
            class_id = int(data[0])
            points = list(map(float, data[1:]))
            
            # OBB 영역 계산
            x_coords = points[0::2]
            y_coords = points[1::2]
            width_px = max(x_coords) - min(x_coords)
            height_px = max(y_coords) - min(y_coords)
            area = width_px * height_px
            
            # 정규화된 값 기준으로 작은 소자만 저장
            if area <= threshold_normalized:
                output_file.write(line)  # 작은 소자만 기록

    
    print(f"Processed {input_path} → {output_path} (Small Objects Only)")

print("✅ 라벨 정제 완료: 작은 소자만 포함된 데이터셋 생성")
