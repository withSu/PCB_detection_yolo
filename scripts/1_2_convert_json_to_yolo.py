import json
import os

# 경로 설정
INPUT_DIR = os.path.join("..", "dataset", "3_new_raw_json")
OUTPUT_DIR = os.path.join("..", "dataset", "4_labels")

# 출력 디렉터리 없으면 생성
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 클래스 매핑 (YOLO 형식은 숫자 클래스 ID를 사용함)
CLASS_NAMES = {
    'component': 0,
}


def convert_to_yolo_obb(input_path, output_path):
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    image_width = data.get('imageWidth')
    image_height = data.get('imageHeight')
    
    if not image_width or not image_height:
        print(f"⚠️ Warning: Missing 'imageWidth' or 'imageHeight' in {input_path}")
        return
    
    with open(output_path, 'w', encoding='utf-8') as out_f:
        for shape in data.get('shapes', []):
            label = shape.get('label', 'unknown')
            points = shape.get('points', [])
            
            # 클래스 ID 변환
            if label not in CLASS_NAMES:
                print(f"⚠️ Warning: Unknown label '{label}' in {input_path}. Skipping...")
                continue
            class_id = CLASS_NAMES[label]
            
            if len(points) == 4:  # 꼭짓점 4개를 가진 OBB
                x_coords = [point[0] / image_width for point in points]
                y_coords = [point[1] / image_height for point in points]
                out_f.write(f"{class_id} " + " ".join([f"{x:.6f} {y:.6f}" for x, y in zip(x_coords, y_coords)]) + "\n")
            
            elif len(points) == 2:  # 두 점만 있는 경우, 사각형으로 변환
                x1, y1 = points[0]
                x2, y2 = points[1]
                rect_points = [
                    [x1 / image_width, y1 / image_height],  # 좌상단
                    [x2 / image_width, y1 / image_height],  # 우상단
                    [x2 / image_width, y2 / image_height],  # 우하단
                    [x1 / image_width, y2 / image_height]   # 좌하단
                ]
                x_coords = [point[0] for point in rect_points]
                y_coords = [point[1] for point in rect_points]
                out_f.write(f"{class_id} " + " ".join([f"{x:.6f} {y:.6f}" for x, y in zip(x_coords, y_coords)]) + "\n")
            else:
                print(f"❌ Unsupported shape with {len(points)} points in {input_path}. Skipping...")


def process_directory():
    json_files = [f for f in os.listdir(INPUT_DIR) if f.endswith(".json")]

    if not json_files:
        print("❗ No JSON files found in the input directory.")
        return

    for file_name in json_files:
        input_path = os.path.join(INPUT_DIR, file_name)
        output_file_name = os.path.splitext(file_name)[0] + ".txt"
        output_path = os.path.join(OUTPUT_DIR, output_file_name)

        convert_to_yolo_obb(input_path, output_path)
        print(f"✅ Converted: {input_path} → {output_path}")

if __name__ == "__main__":
    process_directory()
    print("✅ JSON to YOLO OBB conversion complete!")
