import json
import os
import cv2
import matplotlib.pyplot as plt
import random

def calculate_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[0] + boxA[2], boxB[0] + boxB[2])
    yB = min(boxA[1] + boxA[3], boxB[1] + boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = boxA[2] * boxA[3]
    boxBArea = boxB[2] * boxB[3]
    iou = interArea / float(boxAArea + boxBArea - interArea) if (boxAArea + boxBArea - interArea) > 0 else 0
    return iou

def get_category(area):
    # COCO 기준: small: area < 32×32, medium: 32×32 ≤ area < 96×96, large: area ≥ 96×96
    if area < (32 * 32):
        return 'small'
    elif area < (96 * 96):
        return 'medium'
    else:
        return 'large'

# 데이터 로드
gt_file = "/home/a/A_2024_selfcode/PCB_yolo/dataset/ground_truth.json"
pred_file = "/home/a/A_2024_selfcode/PCB_yolo/outputs_for_exper/run2/coco_predictions_final_normalized.json"
image_dir = "/home/a/A_2024_selfcode/PCB_yolo/dataset/images/val"

with open(gt_file, "r") as f:
    gt_data = json.load(f)

with open(pred_file, "r") as f:
    pred_data = json.load(f)

image_id_to_filename = {img["id"]: img["file_name"] for img in gt_data["images"]}
sample_images = random.sample(list(image_id_to_filename.keys()), min(5, len(image_id_to_filename)))

for image_id in sample_images:
    image_name = image_id_to_filename[image_id]
    image_path = os.path.join(image_dir, image_name)

    img = cv2.imread(image_path)
    if img is None:
        print(f"⚠ Warning: {image_path} 로드 실패. 스킵한다.")
        continue
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_h, img_w, _ = img.shape

    gt_bboxes = [ann["bbox"] for ann in gt_data["annotations"] if ann["image_id"] == image_id]
    pred_bboxes = [pred["bbox"] for pred in pred_data if pred["image_id"] == image_id]

    plt.figure(figsize=(10, 10))
    plt.imshow(img)

    # GT 바운딩 박스 (파란색): 라벨은 표시하지 않음.
    for bbox in gt_bboxes:
        x, y, w, h = bbox
        px = x * img_w
        py = y * img_h
        pw = w * img_w
        ph = h * img_h
        plt.gca().add_patch(
            plt.Rectangle((px, py), pw, ph, edgecolor='blue', linewidth=1, fill=False)
        )

    # 예측 바운딩 박스 (빨간색): 텍스트 라벨에 카테고리와 픽셀 면적 표시 (배경 없음, fontsize=6)
    for bbox in pred_bboxes:
        x, y, w, h = bbox
        px = x * img_w
        py = y * img_h
        pw = w * img_w
        ph = h * img_h
        area = pw * ph
        category = get_category(area)
        plt.gca().add_patch(
            plt.Rectangle((px, py), pw, ph, edgecolor='red', linewidth=1, fill=False)
        )
        plt.text(px + 2, py + 6, f"Pred: {category} ({int(area)})", color="red", fontsize=6)

    # 평균 IoU 계산 (픽셀 단위)
    gt_bboxes_pixel = [(box[0]*img_w, box[1]*img_h, box[2]*img_w, box[3]*img_h) for box in gt_bboxes]
    pred_bboxes_pixel = [(box[0]*img_w, box[1]*img_h, box[2]*img_w, box[3]*img_h) for box in pred_bboxes]
    iou_scores = [calculate_iou(gt_box, pred_box) for gt_box in gt_bboxes_pixel for pred_box in pred_bboxes_pixel]
    avg_iou = sum(iou_scores) / len(iou_scores) if iou_scores else 0

    plt.title(f"{image_name}\nAvg IoU: {avg_iou:.2f}", fontsize=10)
    plt.axis("off")
    plt.show()
