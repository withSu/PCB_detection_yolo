import os
import json

def get_filenames_from_folder(folder_path, extension):
    """
    특정 폴더 내에서 주어진 확장자(extension)를 가진 파일들의 리스트를 반환
    """
    if not os.path.exists(folder_path):
        print(f"⚠ Warning: {folder_path} 폴더가 존재하지 않습니다!")
        return set()
    
    return {os.path.splitext(f)[0] for f in os.listdir(folder_path) if f.endswith(extension)}

def check_missing_files(labels_dir, images_dir, gt_file, pred_file):
    """
    labels/val, images/val, ground_truth.json, coco_predictions.json에서 파일 이름이 일치하는지 확인
    """
    # YOLO 라벨 파일(.txt) → 이미지 파일명(.jpg) 변환
    label_files = get_filenames_from_folder(labels_dir, ".txt")
    
    # 이미지 파일 목록 가져오기
    image_files = get_filenames_from_folder(images_dir, ".jpg")

    # GT 파일에서 참조하는 이미지 목록 가져오기
    with open(gt_file, "r") as f:
        gt_data = json.load(f)
    gt_files = {os.path.splitext(img["file_name"])[0] for img in gt_data["images"]}

    # YOLO 예측 파일에서 참조하는 이미지 목록 가져오기
    with open(pred_file, "r") as f:
        pred_data = json.load(f)
    pred_files = {pred["image_id"] for pred in pred_data}

    # 누락된 파일 확인
    missing_in_images = label_files - image_files  # 라벨은 있는데 이미지가 없는 경우
    missing_in_labels = image_files - label_files  # 이미지는 있는데 라벨이 없는 경우
    missing_in_gt = image_files - gt_files  # 이미지가 GT에 등록되지 않은 경우
    missing_from_gt = gt_files - image_files  # GT에만 있고 실제 파일은 없는 경우
    missing_in_pred = gt_files - pred_files  # GT에 있는데 YOLO 예측에는 없는 경우
    missing_from_pred = pred_files - gt_files  # YOLO 예측에는 있는데 GT에 없는 경우

    # 결과 출력
    print(f"✅ 총 {len(image_files)}개의 이미지 파일, {len(label_files)}개의 라벨 파일, {len(gt_files)}개의 GT 파일, {len(pred_files)}개의 예측 파일이 존재합니다.\n")
    
    if missing_in_images:
        print(f"⚠ Warning: {len(missing_in_images)}개의 라벨 파일이 있지만 이미지가 없습니다:")
        print(sorted(missing_in_images))

    if missing_in_labels:
        print(f"⚠ Warning: {len(missing_in_labels)}개의 이미지 파일이 있지만 라벨이 없습니다:")
        print(sorted(missing_in_labels))

    if missing_in_gt:
        print(f"⚠ Warning: {len(missing_in_gt)}개의 이미지 파일이 GT에 등록되지 않았습니다:")
        print(sorted(missing_in_gt))

    if missing_from_gt:
        print(f"⚠ Warning: {len(missing_from_gt)}개의 GT 등록 이미지가 실제로 존재하지 않습니다:")
        print(sorted(missing_from_gt))

    if missing_in_pred:
        print(f"⚠ Warning: {len(missing_in_pred)}개의 GT 등록 이미지가 YOLO 예측에 없습니다:")
        print(sorted(missing_in_pred))

    if missing_from_pred:
        print(f"⚠ Warning: {len(missing_from_pred)}개의 YOLO 예측 이미지가 GT에 없습니다:")
        print(sorted(missing_from_pred))

    if not (missing_in_images or missing_in_labels or missing_in_gt or missing_from_gt or missing_in_pred or missing_from_pred):
        print("✅ 모든 파일이 정상적으로 일치합니다!")

# 실행
if __name__ == "__main__":
    labels_dir = "/home/a/A_2024_selfcode/PCB_yolo/dataset/labels/val"
    images_dir = "/home/a/A_2024_selfcode/PCB_yolo/dataset/images/val"
    gt_file = "/home/a/A_2024_selfcode/PCB_yolo/dataset/ground_truth.json"
    pred_file = "/home/a/A_2024_selfcode/PCB_yolo/outputs_for_exper/run/coco_predictions.json"

    check_missing_files(labels_dir, images_dir, gt_file, pred_file)
