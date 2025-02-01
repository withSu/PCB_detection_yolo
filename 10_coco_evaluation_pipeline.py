from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

def coco_evaluation(gt_file, dt_file):
    """
    COCO AP/AR 평가 수행 (IoU 기준 [0.50, 0.75, 0.95] 유지)
    :param gt_file: 변환된 GT 파일 (픽셀 단위)
    :param dt_file: 변환된 예측 파일 (픽셀 단위)
    """
    # GT 및 예측 데이터 로드
    coco_gt = COCO(gt_file)
    coco_dt = coco_gt.loadRes(dt_file)

    # COCO 평가 수행 (IoU 기준 [0.50, 0.75, 0.95])
    coco_eval = COCOeval(coco_gt, coco_dt, "bbox")
    coco_eval.params.iouThrs = [0.50, 0.75, 0.95]  # IoU 임계값 유지
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    return coco_eval.stats  # AP, AR 값 리턴

if __name__ == "__main__":
    gt_path = "/home/a/A_2024_selfcode/PCB_yolo/dataset/ground_truth_pixel.json"  # 변환된 GT 파일 (픽셀 단위)
    dt_path = "/home/a/A_2024_selfcode/PCB_yolo/outputs_for_exper/run/coco_predictions_final.json"  # 변환된 예측 파일 (픽셀 단위)

    results = coco_evaluation(gt_path, dt_path)
    print(f"COCO Evaluation Results: {results}")
