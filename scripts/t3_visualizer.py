import json
import cv2
import os
import numpy as np

def process_and_visualize(data_dir='merged_data', output_dir='merged_output'):
    # output 폴더 생성
    os.makedirs(output_dir, exist_ok=True)

    # JSON 파일들을 찾아서 처리
    for filename in os.listdir(data_dir):
        if filename.endswith('.json'):
            base_name = os.path.splitext(filename)[0]
            json_path = os.path.join(data_dir, filename)

            for ext in ['.jpg', '.png', '.jpeg']:
                image_path = os.path.join(data_dir, base_name + ext)
                if not os.path.exists(image_path):
                    continue

                # 이미지 로드
                image = cv2.imread(image_path)
                if image is None:
                    print(f"이미지를 불러올 수 없습니다: {image_path}")
                    break

                # JSON 파일 로드
                with open(json_path, 'r') as f:
                    data = json.load(f)

                # 각 라벨에 대한 색상 매핑 (랜덤 색상)
                label_colors = {}

                # shapes 리스트를 순회하면서 바운딩 박스 그리기
                for shape in data['shapes']:
                    label = shape['label']
                    points = shape['points']
                    
                    # 라벨에 대한 색상이 없으면 새로 생성
                    if label not in label_colors:
                        label_colors[label] = tuple(np.random.randint(0, 255, 3).tolist())
                    
                    # polygon인 경우 모든 점을 연결
                    if shape['shape_type'] == 'polygon':
                        points = np.array(points, np.int32)
                        cv2.polylines(image, [points], True, label_colors[label], 2)
                        x1, y1 = points[0]
                    else:  # rectangle인 경우
                        x1, y1 = map(int, points[0])
                        x2, y2 = map(int, points[1])
                        cv2.rectangle(image, (x1, y1), (x2, y2), label_colors[label], 2)
                    
                    # 라벨 텍스트 추가
                    cv2.putText(image, label, (int(x1), int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 
                              0.5, label_colors[label], 2)

                # 결과 이미지를 output 폴더에 저장
                output_filename = os.path.basename(os.path.splitext(image_path)[0]) + '.jpg'
                output_path = os.path.join(output_dir, output_filename)
                cv2.imwrite(output_path, image)
                print(f"시각화된 이미지가 저장되었습니다: {output_path}")
                break

if __name__ == "__main__":
    process_and_visualize('/home/a/Desktop/MirTech_Seg_2024/1119/파이썬코드/result_confidence/merge', '/home/a/Desktop/MirTech_Seg_2024/1119/파이썬코드/result_confidence/visualizer')
