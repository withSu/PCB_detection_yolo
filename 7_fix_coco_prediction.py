# fix_predictions.py

import json
import os

def fix_image_id(gt_file, pred_file, output_file):
    """
    YOLO 예측 결과(`coco_predictions.json`)의 `image_id`를 GT(`ground_truth.json`)의 `image_id`와 동일하게 변경한다.
    """
    # GT 파일 로드
    with open(gt_file, "r") as f:
        gt_data = json.load(f)

    # 이미지 파일명 → image_id 매핑 생성 (확장자 제거)
    image_id_map = {os.path.splitext(img["file_name"])[0]: img["id"] for img in gt_data["images"]}

    # 예측 파일 로드
    with open(pred_file, "r") as f:
        pred_data = json.load(f)

    fixed_preds = []
    for pred in pred_data:
        file_name = pred["image_id"]
        
        # GT에서 해당 이미지 ID 찾기
        if file_name in image_id_map:
            pred["image_id"] = image_id_map[file_name]  # 숫자로 변경
            fixed_preds.append(pred)
        else:
            print(f"⚠ Warning: `{file_name}`이(가) GT에 없음. 제거됨.")

    # 수정된 예측 결과 저장
    with open(output_file, "w") as f:
        json.dump(fixed_preds, f, indent=4)

    print(f"✅ `image_id` 수정 완료: {output_file}")

def fix_category_id(gt_file, pred_file, output_file):
    """
    COCO 평가를 위해 `coco_predictions_fixed.json`의 `category_id`를 `ground_truth.json`과 동일하게 변경한다.
    """
    # GT 파일 로드
    with open(gt_file, "r") as f:
        gt_data = json.load(f)

    # GT의 category_id 가져오기 (첫 번째 카테고리 기준)
    gt_category_id = gt_data["categories"][0]["id"] if gt_data["categories"] else 0

    # 예측 파일 로드
    with open(pred_file, "r") as f:
        pred_data = json.load(f)

    # `category_id` 수정
    for pred in pred_data:
        pred["category_id"] = gt_category_id

    # 수정된 예측 결과 저장
    with open(output_file, "w") as f:
        json.dump(pred_data, f, indent=4)

    print(f"✅ `category_id` 수정 완료: {output_file}")

def normalize_bbox(gt_file, pred_file, output_file, img_width=3904, img_height=3904):
    """
    COCO 평가를 위해 `coco_predictions_final.json`의 `bbox`를 `ground_truth.json`과 동일한 형식으로 변환한다.
    """
    # GT 파일 로드
    with open(gt_file, "r") as f:
        gt_data = json.load(f)

    # GT `bbox`가 정규화된 값인지 확인
    bbox_normalized = all(0 <= ann["bbox"][2] <= 1 and 0 <= ann["bbox"][3] <= 1 for ann in gt_data["annotations"])

    # 예측 파일 로드
    with open(pred_file, "r") as f:
        pred_data = json.load(f)

    # `bbox` 좌표 변환
    for pred in pred_data:
        x, y, w, h = pred["bbox"]

        if bbox_normalized:
            # GT가 정규화된 경우, 예측값도 정규화
            x /= img_width
            y /= img_height
            w /= img_width
            h /= img_height

            # 정규화 후 좌표 값이 1을 초과하지 않도록 제한
            x = min(x, 1)
            y = min(y, 1)
            w = min(w, 1 - x)
            h = min(h, 1 - y)

        pred["bbox"] = [x, y, w, h]

    # 수정된 예측 결과 저장
    with open(output_file, "w") as f:
        json.dump(pred_data, f, indent=4)

    print(f"✅ `bbox` 형식 수정 완료: {output_file}")

if __name__ == "__main__":
    # 1단계: image_id 수정
    gt_path = "dataset/ground_truth.json"
    pred_path = "outputs_for_exper/run/coco_predictions.json"
    output_path = "outputs_for_exper/run/coco_predictions_fixed.json"
    fix_image_id(gt_path, pred_path, output_path)

    # 2단계: category_id 수정
    gt_path = "dataset/ground_truth.json"
    pred_path = "outputs_for_exper/run/coco_predictions_fixed.json"
    output_path = "outputs_for_exper/run/coco_predictions_final.json"
    fix_category_id(gt_path, pred_path, output_path)

    # 3단계: bbox 정규화
    gt_path = "dataset/ground_truth.json"
    pred_path = "outputs_for_exper/run/coco_predictions_final.json"
    output_path = "outputs_for_exper/run/coco_predictions_final_normalized.json"
    normalize_bbox(gt_path, pred_path, output_path)
