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

def convert_to_yolo(input_path, output_path):
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
            shape_type = shape.get('shape_type')
            points = shape.get('points', [])
            
            # 클래스 ID 변환
            if label not in CLASS_NAMES:
                print(f"⚠️ Warning: Unknown label '{label}' in {input_path}. Skipping...")
                continue
            class_id = CLASS_NAMES[label]
            
            if shape_type == 'polygon':
                x_coords = [point[0] for point in points]
                y_coords = [point[1] for point in points]
                
                x_min = min(x_coords)
                x_max = max(x_coords)
                y_min = min(y_coords)
                y_max = max(y_coords)
                
                x_center = (x_min + x_max) / 2 / image_width
                y_center = (y_min + y_max) / 2 / image_height
                width = (x_max - x_min) / image_width
                height = (y_max - y_min) / image_height
                
                out_f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
            
            elif shape_type == 'rectangle':
                x_min = points[0][0]
                y_min = points[0][1]
                x_max = points[1][0]
                y_max = points[1][1]
                
                x_center = (x_min + x_max) / 2 / image_width
                y_center = (y_min + y_max) / 2 / image_height
                width = (x_max - x_min) / image_width
                height = (y_max - y_min) / image_height
                
                out_f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
            
            else:
                print(f"❌ Unsupported shape_type: '{shape_type}' in {input_path}. Skipping...")

def process_directory():
    json_files = [f for f in os.listdir(INPUT_DIR) if f.endswith(".json")]
    
    if not json_files:
        print("❗ No JSON files found in the input directory.")
        return
    
    for file_name in json_files:
        input_path = os.path.join(INPUT_DIR, file_name)
        output_file_name = os.path.splitext(file_name)[0] + ".txt"
        output_path = os.path.join(OUTPUT_DIR, output_file_name)
        
        convert_to_yolo(input_path, output_path)
        print(f"✅ Converted: {input_path} → {output_path}")

if __name__ == "__main__":
    process_directory()
    print("✅ JSON to YOLO conversion complete!")
