import os
import json

def convert_yolo_obb_to_coco(yolo_labels_dir, coco_output_file, image_dir, class_names, img_width=3904, img_height=3904, normalize=False):
    """
    OBB(Oriented Bounding Box) YOLO 라벨(txt) -> COCO JSON 변환 (GT 데이터)
    `normalize=True`로 설정하면 bbox를 (0~1) 정규화된 좌표로 변환
    """
    coco_data = {
        "images": [],
        "annotations": [],
        "categories": [{"id": i, "name": name} for i, name in enumerate(class_names)]
    }

    annotation_id = 1
    image_id = 1

    for label_file in sorted(os.listdir(yolo_labels_dir)):
        if not label_file.endswith(".txt"):
            continue

        img_name = label_file.replace(".txt", ".jpg")  # 이미지 파일명 추정
        img_path = os.path.join(image_dir, img_name)

        if not os.path.exists(img_path):
            print(f"⚠ Warning: 이미지 {img_path} 없음. 건너뜀.")
            continue

        coco_data["images"].append({
            "id": image_id,
            "file_name": img_name,
            "width": img_width,  
            "height": img_height
        })

        with open(os.path.join(yolo_labels_dir, label_file), "r") as f:
            lines = f.readlines()

        for line in lines:
            parts = line.strip().split()
            cls_id = int(parts[0])
            coords = list(map(float, parts[1:]))  # 8개 좌표 가져오기

            if len(coords) != 8:
                print(f"⚠ Warning: {label_file} 라벨 데이터 오류 (좌표 개수 불일치)")
                continue

            # OBB 좌표 정규화 → COCO bbox 변환
            x1, y1, x2, y2, x3, y3, x4, y4 = coords
            
            # YOLO 정규화 좌표 (0~1) → 픽셀 좌표 변환
            x1, y1 = x1 * img_width, y1 * img_height
            x2, y2 = x2 * img_width, y2 * img_height
            x3, y3 = x3 * img_width, y3 * img_height
            x4, y4 = x4 * img_width, y4 * img_height

            # COCO 형식으로 변환 (x_min, y_min, w, h)
            x_min = min(x1, x2, x3, x4)
            y_min = min(y1, y2, y3, y4)
            x_max = max(x1, x2, x3, x4)
            y_max = max(y1, y2, y3, y4)

            width = x_max - x_min
            height = y_max - y_min

            if normalize:
                x_min /= img_width
                y_min /= img_height
                width /= img_width
                height /= img_height

            bbox = [x_min, y_min, width, height]  # [x, y, width, height]

            coco_data["annotations"].append({
                "id": annotation_id,
                "image_id": image_id,
                "category_id": cls_id,
                "bbox": bbox,
                "area": width * height,  # 면적 계산
                "iscrowd": 0
            })
            annotation_id += 1

        image_id += 1

    with open(coco_output_file, "w") as f:
        json.dump(coco_data, f, indent=4)

    print(f"✅ GT COCO JSON 변환 완료: {coco_output_file}")

# 변환 실행
if __name__ == "__main__":
    yolo_labels_dir = "/home/a/A_2024_selfcode/PCB_yolo/dataset/labels/val"  # YOLO 검증 데이터 라벨 폴더
    output_json = "/home/a/A_2024_selfcode/PCB_yolo/dataset/ground_truth.json"
    image_dir = "/home/a/A_2024_selfcode/PCB_yolo/dataset/images/val"  # 검증 이미지 폴더
    class_names = ["component"]  # 데이터셋 클래스

    # normalize=True로 설정하면 bbox 좌표를 정규화 (0~1)
    convert_yolo_obb_to_coco(yolo_labels_dir, output_json, image_dir, class_names, normalize=True)
