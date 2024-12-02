# Load AutoEncoder

import os, sys, time
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2

# 현재 프로젝트의 경로 활용
cwd = os.path.dirname(os.path.abspath(__file__))

# 차원축소 데이터 저장 경로 지정
npyResDir = "./auto_npyRes/fog_auto_M_32/"
os.makedirs(npyResDir, exist_ok=True)

# 재귀함수 활용하여 결정된 데이터셋의 절대경로 및 용량을 구해 출력함.
def get_dataset_paths():
    def get_folder_size(path):
        total_size = 0
        for dirpath, dirnames, filenames in os.walk(path):
            for file in filenames:
                filepath = os.path.join(dirpath, file)
                total_size += os.path.getsize(filepath) # 파일 크기를 누적
        return total_size

    if len(sys.argv) != 2:
        print("\n" + "Usage: python script_name.py <choice> (1, 2)")
        return None, None, None

    choice = sys.argv[1]

    if choice == "1":
        train_path = os.path.abspath(os.path.join(cwd, "../_Dataset/cat_dog/training_set"))
        test_path = os.path.abspath(os.path.join(cwd, "../_Dataset/cat_dog/test_set"))
    elif choice == "2":
        train_path = os.path.abspath(os.path.join(cwd, "../_Dataset/FER_2013/training_set"))
        test_path = os.path.abspath(os.path.join(cwd, "../_Dataset/FER_2013/test_set"))
    else:
        print("Invalid choice. Please select 1, 2.")
        return None, None, None

    print("\n\n" + "=" * 60)
    print("--- DataSet Selection ---")
    print(f"Training Path: {train_path}")
    print(f"Testing Path: {test_path}")
    training_folder_size = get_folder_size(train_path) # 폴더 용량 계산
    print(f"Training folder size: {training_folder_size / (1024**2):.2f} MB\n" + "="*60 + "\n")

    return train_path, test_path, choice

# 함수 호출 및 결과 출력
train_path, test_path, choice = get_dataset_paths()
if train_path == None:
    exit(1)

""" AutoEncoder 시간 측정 시작 """
fog_auto_start_time = time.time()

# ImageDataGenerator 생성
datagen = ImageDataGenerator(rescale=1.0 / 255.0)
BATCH_SIZE = 32
img_height, img_width = 224, 224

# 데이터 로드
training_set = datagen.flow_from_directory(
    train_path,
    target_size=(img_height, img_width),
    batch_size=BATCH_SIZE,
    shuffle=False,
    class_mode='categorical'
)

test_set = datagen.flow_from_directory(
    test_path,
    target_size=(img_height, img_width),
    batch_size=BATCH_SIZE,
    shuffle=False,
    class_mode='categorical'
)

# MobileNetV2를 이용한 특성 추출
mobilenet = MobileNetV2(weights='imagenet', include_top=False, input_shape=(img_height, img_width, 3))
for layer in mobilenet.layers:
    layer.trainable = False

train_features = mobilenet.predict(training_set)
test_features = mobilenet.predict(test_set)

# 저장된 Encoder 불러오기
# Encoder를 사용하여 차원 축소
if choice == "1":
    encoder = load_model(npyResDir + "fog_autoencoder_mobilenet_32x32_dogcat.h5")
    train_features_reduced = encoder.predict(train_features)
    test_features_reduced = encoder.predict(test_features)
elif choice == "3":
    encoder = load_model(npyResDir + "fog_autoencoder_mobilenet_32x32_FEB2013.h5")
    train_features_reduced = encoder.predict(train_features)
    test_features_reduced = encoder.predict(test_features)
print("Encoder model Loaded.")

def measure_npy_file_sizes(directory):
    # 차원 축소된 데이터 저장
    print("\n\n" + "=" * 60)
    print(f"--- reSized demention files ---")
    np.save(npyResDir + "train_features_pca.npy", train_features_reduced)
    np.save(npyResDir + "test_features_pca.npy", test_features_reduced)
    np.save(npyResDir + "train_labels.npy", training_set.classes)
    np.save(npyResDir + "test_labels.npy", test_set.classes)
    print(f" -> PCA data saved at {npyResDir}\n" + "="*60)
    print(f"--- File sizes in {directory} ---\n")

    total_size = 0
    # 디렉토리의 모든 파일 확인
    for file_name in os.listdir(directory):
        file_path = os.path.join(directory, file_name)
        if os.path.isfile(file_path):  # 파일인지 확인
            file_size = os.path.getsize(file_path)  # 파일 크기 (바이트)
            print(f" - {file_name}: {file_size / (1024 ** 2):.2f} MB")  # MB 단위로 출력
            total_size += file_size

    # 총 용량 출력
    print(f"\nTotal size of all files: {total_size / (1024 ** 2):.2f} MB\n" + "="*60 + "\n")
# npyResDir 내 파일 용량 측정 함수 호출
measure_npy_file_sizes(npyResDir)

""" Fogging+AutoEncoder 시간 측정 종료 """
fog_auto_end_time = time.time()
fog_auto_total_time = fog_auto_end_time - fog_auto_start_time

# 시간 산정
fog_pca_total_time = fog_auto_end_time - fog_auto_start_time
print("\n\n" + "=" * 60)
print("--- Fogging + (load)AutoEncoder (MobileNet) | Size(16x16) TimeSet ---")
print(f"* Total execute time : {fog_auto_total_time:.3f} seconds.")
print("="*60)