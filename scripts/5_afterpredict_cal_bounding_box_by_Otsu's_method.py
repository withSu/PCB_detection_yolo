import os
import numpy as np
from skimage.filters import threshold_otsu
import matplotlib.pyplot as plt
from collections import Counter

# 1. 라벨 파일에서 Bounding Box 데이터 추출
labels_dir = "runs/detect/predict2/labels"  # 라벨 파일 경로
IMAGE_WIDTH = 1024  # 이미지 가로 크기
IMAGE_HEIGHT = 1024  # 이미지 세로 크기

bounding_box_data = []  # Bounding Box 데이터를 저장할 리스트

# 모든 라벨 파일 처리
for file_name in os.listdir(labels_dir):
    file_path = os.path.join(labels_dir, file_name)
    with open(file_path, 'r') as file:
        for line in file.readlines():
            class_id, x_center, y_center, width, height = map(float, line.split())
            width_px = width * IMAGE_WIDTH  # 가로 픽셀 크기
            height_px = height * IMAGE_HEIGHT  # 세로 픽셀 크기
            area = width_px * height_px  # Bounding Box 면적
            bounding_box_data.append({"file": file_name, "class_id": int(class_id), "area": area})

# 2. Bounding Box 면적만 추출
areas = [entry["area"] for entry in bounding_box_data]

# 3. Otsu's Method로 임계값 계산
threshold = threshold_otsu(np.array(areas))
print(f"Otsu's Threshold: {threshold:.2f}")

# 4. 작은 소자와 큰 소자 분류
small_objects = [entry for entry in bounding_box_data if entry["area"] <= threshold]
large_objects = [entry for entry in bounding_box_data if entry["area"] > threshold]

print(f"Small Objects: {len(small_objects)}, Large Objects: {len(large_objects)}")

# 5. 클래스별 작은 소자와 큰 소자 분포
small_class_count = Counter([obj["class_id"] for obj in small_objects])
large_class_count = Counter([obj["class_id"] for obj in large_objects])

# 터미널에 클래스별 분포 출력
print("Small Object Class Distribution:")
for class_id, count in small_class_count.items():
    print(f"  Class {class_id}: {count}")

print("Large Object Class Distribution:")
for class_id, count in large_class_count.items():
    print(f"  Class {class_id}: {count}")

# 6. 히스토그램 시각화 (x축 범위 제한 및 로그 스케일 추가)
max_display_area = min(max(areas), threshold * 3)  # x축 범위를 제한
plt.figure(figsize=(12, 6))
plt.hist(
    areas, bins=50, color='blue', alpha=0.7, label="Bounding Box Areas", range=(0, max_display_area)
)
plt.axvline(x=threshold, color='red', linestyle='--', label=f"Otsu's Threshold ({threshold:.2f} px²)")
plt.title("Histogram of Bounding Box Areas (Zoomed)")
plt.xlabel("Area (pixels²)")  # x축 단위 추가
plt.ylabel("Frequency (log scale)")  # y축 단위 명시
plt.yscale("log")  # 로그 스케일 적용
plt.legend()
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.show()

# 7. 클래스별 빈도수 시각화 (막대그래프)
plt.figure(figsize=(12, 6))
plt.bar(small_class_count.keys(), small_class_count.values(), color="orange", alpha=0.8, label="Small Objects")
plt.bar(large_class_count.keys(), large_class_count.values(), color="green", alpha=0.8, label="Large Objects")
plt.title("Frequency of Objects per Class")
plt.xlabel("Class ID")
plt.ylabel("Frequency (count)")
plt.legend()
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.xticks(range(max(max(small_class_count.keys()), max(large_class_count.keys())) + 1))  # 클래스 ID 표시
plt.show()

# 8. 클래스별 면적 합 시각화 (막대그래프)
small_class_area = Counter()
large_class_area = Counter()

for obj in small_objects:
    small_class_area[obj["class_id"]] += obj["area"]

for obj in large_objects:
    large_class_area[obj["class_id"]] += obj["area"]

plt.figure(figsize=(12, 6))
plt.bar(small_class_area.keys(), small_class_area.values(), color="orange", alpha=0.8, label="Small Objects")
plt.bar(large_class_area.keys(), large_class_area.values(), color="green", alpha=0.8, label="Large Objects")
plt.title("Total Area per Class")
plt.xlabel("Class ID")
plt.ylabel("Total Area (pixels²)")
plt.legend()
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.xticks(range(max(max(small_class_area.keys()), max(large_class_area.keys())) + 1))  # 클래스 ID 표시
plt.show()

# 9. Threshold에 따른 작은/큰 소자 구분 시각화
small_areas = [area for area in areas if area <= threshold]
large_areas = [area for area in areas if area > threshold]

plt.figure(figsize=(12, 6))
plt.hist(small_areas, bins=50, color='orange', alpha=0.7, label="Small Objects", range=(0, max_display_area))
plt.hist(large_areas, bins=50, color='green', alpha=0.7, label="Large Objects", range=(0, max_display_area))
plt.axvline(x=threshold, color='red', linestyle='--', label=f"Otsu's Threshold ({threshold:.2f} px²)")
plt.title("Bounding Box Areas Separated by Threshold")
plt.xlabel("Area (pixels²)")
plt.ylabel("Frequency (log scale)")
plt.yscale("log")
plt.legend()
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.show()
