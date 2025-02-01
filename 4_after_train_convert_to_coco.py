import json
import os

def convert_yolo_to_coco(yolo_pred_file, coco_output_file):
    """
    YOLO OBB 예측 결과 (rbox) -> COCO 평가 JSON 형식으로 변환
    """
    with open(yolo_pred_file, "r") as f:
        yolo_preds = json.load(f)

    coco_results = []

    for pred in yolo_preds:
        image_id = pred["image_id"]
        category_id = pred["category_id"]
        score = pred["score"]
        rbox = pred["rbox"]  # YOLO는 회전된 박스 형식 사용

        # COCO 평가용 bbox 변환 (xywh)
        x_center, y_center, w, h, theta = rbox  # 회전 정보 포함
        x = x_center - (w / 2)  # COCO는 좌상단 기준 x
        y = y_center - (h / 2)  # COCO는 좌상단 기준 y

        coco_results.append({
            "image_id": image_id,
            "category_id": category_id,
            "bbox": [x, y, w, h],  # COCO bbox 형식
            "score": score
        })

    with open(coco_output_file, "w") as f:
        json.dump(coco_results, f, indent=4)

    print(f"✅ COCO 평가용 JSON 변환 완료: {coco_output_file}")

# 변환 실행
if __name__ == "__main__":
    yolo_pred_file = "/home/a/A_2024_selfcode/PCB_yolo/outputs_for_exper/run/predictions.json"
    coco_output_file = "/home/a/A_2024_selfcode/PCB_yolo/outputs_for_exper/run/coco_predictions.json"

    convert_yolo_to_coco(yolo_pred_file, coco_output_file)
