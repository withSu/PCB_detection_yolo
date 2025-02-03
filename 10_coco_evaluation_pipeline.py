import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

def coco_evaluation(gt_file, dt_file):
    """
    COCO AP/AR 평가를 COCO 공식 기준 (IoU 0.50~0.95, 0.05 간격)으로 수행한다.
    :param gt_file: 픽셀 단위로 변환된 GT 파일 경로이다.
    :param dt_file: 픽셀 단위로 변환된 예측 파일 경로이다.
    """
    # GT와 예측 데이터를 COCO 형식으로 로드한다.
    coco_gt = COCO(gt_file)
    coco_dt = coco_gt.loadRes(dt_file)
    
    # COCOeval 객체를 생성하고, IoU 임계값을 공식 기준에 맞게 설정한다.
    coco_eval = COCOeval(coco_gt, coco_dt, "bbox")
    coco_eval.params.iouThrs = np.linspace(0.5, 0.95, 10)
    
    # 평가 과정을 실행한다.
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    
    return coco_eval.stats

if __name__ == "__main__":
    gt_path = "/home/a/A_2024_selfcode/PCB_yolo/dataset/ground_truth_pixel.json"
    dt_path = "/home/a/A_2024_selfcode/PCB_yolo/outputs_for_exper/run2/coco_predictions_final.json"
    
    results = coco_evaluation(gt_path, dt_path)
    print("COCO Evaluation Results:", results)
