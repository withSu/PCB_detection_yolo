from ultralytics import YOLO

# 모델 로드
model = YOLO('/home/a/A_2024_selfcode/PCB/runs/obb/train28/weights/best.pt')

# 이미지 추론
results = model.predict(
    source='/home/a/A_2024_selfcode/PCB/test_images',  # 추론할 이미지 경로
    save=True,                                        # 결과 저장 여부
    conf=0.25                                         # Confidence threshold
)

# 결과 출력
for result in results:
    print(result.boxes)
