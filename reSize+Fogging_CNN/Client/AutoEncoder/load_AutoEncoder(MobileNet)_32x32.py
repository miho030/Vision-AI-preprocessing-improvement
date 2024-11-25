# Load AutoEncoder

import os, time
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2

# 데이터 경로
train_path = os.path.abspath("../../../_Dataset/cat_dog/training_set")
test_path = os.path.abspath("../../../_Dataset/cat_dog/test_set")
npyResDir = "./auto_npyRes/"

os.makedirs(npyResDir, exist_ok=True)

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
    class_mode='binary'
)

test_set = datagen.flow_from_directory(
    test_path,
    target_size=(img_height, img_width),
    batch_size=BATCH_SIZE,
    shuffle=False,
    class_mode='binary'
)

# MobileNetV2를 이용한 특성 추출
mobilenet = MobileNetV2(weights='imagenet', include_top=False, input_shape=(img_height, img_width, 3))
for layer in mobilenet.layers:
    layer.trainable = False

train_features = mobilenet.predict(training_set)
test_features = mobilenet.predict(test_set)

# 저장된 Encoder 불러오기
encoder = load_model(npyResDir + "encoder_32x32_model.h5")
print("Encoder model Loaded.")

# Encoder를 사용하여 차원 축소
train_features_reduced = encoder.predict(train_features)
test_features_reduced = encoder.predict(test_features)

# 차원 축소된 데이터 저장
np.save(npyResDir + "train_features_reduced.npy", train_features_reduced)
np.save(npyResDir + "test_features_reduced.npy", test_features_reduced)
np.save(npyResDir + "train_labels.npy", training_set.classes)
np.save(npyResDir + "test_labels.npy", test_set.classes)

print("Autoencoder data saved.")

""" Fogging+AutoEncoder 시간 측정 종료 """
fog_auto_end_time = time.time()
fog_auto_total_time = fog_auto_end_time - fog_auto_start_time

# 시간 산정
fog_pca_total_time = fog_auto_end_time - fog_auto_start_time
print("\n\n" + "=" * 60)
print("--- Fogging + AutoEncoder TimeSet ---")
print(f"* Total execute time : {fog_auto_total_time:.3f} seconds.")
print("="*60)