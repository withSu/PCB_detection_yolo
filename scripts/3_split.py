import os
import shutil
import random

# 📁 **경로 설정**
DATASET_DIR = '../dataset'
IMAGES_DIR = os.path.join(DATASET_DIR, '1_images')
LABELS_DIR = os.path.join(DATASET_DIR, '4_labels')
OUTPUT_DIR = DATASET_DIR

# YOLO 요구 구조
OUTPUT_IMAGES_TRAIN = os.path.join(OUTPUT_DIR, 'images/train')
OUTPUT_IMAGES_VAL = os.path.join(OUTPUT_DIR, 'images/val')
OUTPUT_LABELS_TRAIN = os.path.join(OUTPUT_DIR, 'labels/train')
OUTPUT_LABELS_VAL = os.path.join(OUTPUT_DIR, 'labels/val')

# ⚖️ **비율 설정**
TRAIN_RATIO = 0.8

# 📂 **디렉터리 생성**
for dir in [OUTPUT_IMAGES_TRAIN, OUTPUT_IMAGES_VAL, OUTPUT_LABELS_TRAIN, OUTPUT_LABELS_VAL]:
    os.makedirs(dir, exist_ok=True)

# 🔍 **이미지와 라벨 파일 매칭**
image_files = set(os.path.splitext(f)[0] for f in os.listdir(IMAGES_DIR) if f.endswith(('.jpg', '.png', '.jpeg')))
label_files = set(os.path.splitext(f)[0] for f in os.listdir(LABELS_DIR) if f.endswith('.txt'))

# 📝 **매칭된 파일만 사용**
matched_files = image_files & label_files

if not matched_files:
    raise ValueError("⚠️ 이미지와 라벨이 매칭된 파일이 없습니다. 파일 이름을 확인해주세요.")

# 🔄 **데이터 분할**
matched_files = list(matched_files)
random.shuffle(matched_files)

train_count = int(len(matched_files) * TRAIN_RATIO)
train_files = matched_files[:train_count]
val_files = matched_files[train_count:]

# 📥 **파일 복사 함수**
def copy_files(files, image_dst, label_dst):
    for file in files:
        image_src = os.path.join(IMAGES_DIR, f"{file}.jpg")
        label_src = os.path.join(LABELS_DIR, f"{file}.txt")

        if os.path.exists(image_src) and os.path.exists(label_src):
            shutil.copy(image_src, os.path.join(image_dst, f"{file}.jpg"))
            shutil.copy(label_src, os.path.join(label_dst, f"{file}.txt"))
        else:
            print(f"⚠️ 파일이 존재하지 않습니다: {file}")

# 🚀 **파일 분할 및 복사 실행**
copy_files(train_files, OUTPUT_IMAGES_TRAIN, OUTPUT_LABELS_TRAIN)
copy_files(val_files, OUTPUT_IMAGES_VAL, OUTPUT_LABELS_VAL)

# 📝 **dataset.yaml 파일 생성**
dataset_yaml_path = os.path.join(DATASET_DIR, 'dataset.yaml')
with open(dataset_yaml_path, 'w') as yaml_file:
    yaml_file.write(f"""train: {os.path.abspath(OUTPUT_IMAGES_TRAIN)}
val: {os.path.abspath(OUTPUT_IMAGES_VAL)}

nc: 11
names: [
  'Chip',
  'CSolder',
  '2sideIC',
  'SOD',
  'Circle',
  '4sideIC',
  'Tantalum',
  'BGA',
  'MELF',
  'Crystal',
  'Array'
]
""")

print(f"✅ 데이터셋 분할 및 YAML 파일 생성 완료")
print(f" - 학습 데이터: {len(train_files)}개")
print(f" - 검증 데이터: {len(val_files)}개")
print(f" - dataset.yaml: {dataset_yaml_path}")
