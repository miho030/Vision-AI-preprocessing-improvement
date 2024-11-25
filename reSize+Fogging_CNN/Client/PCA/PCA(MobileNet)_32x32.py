# PCA w.MobileNet

import os, time
import numpy as np
from sklearn.decomposition import PCA
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2

train_path = os.path.abspath("../../../_Dataset/cat_dog/training_set")
test_path = os.path.abspath("../../../_Dataset/cat_dog/test_set")
npyResDir = "./pca_npyRes/"

os.makedirs(npyResDir, exist_ok=True)

""" Fogging+PCA 전처리 시간 측정 시작 """
fog_pca_start_time = time.time()

# ImageDataGenerator 생성
datagen = ImageDataGenerator(rescale=1.0 / 255.0)
BATCH_SIZE = 32
img_height, img_width = 224, 224

# 훈련 데이터 로드
training_set = datagen.flow_from_directory(
    train_path,
    target_size=(img_height, img_width),
    batch_size=BATCH_SIZE,
    shuffle=False,
    class_mode='binary'
)

# 테스트 데이터 로드
test_set = datagen.flow_from_directory(
    test_path,
    target_size=(img_height, img_width),
    batch_size=BATCH_SIZE,
    shuffle=False,
    class_mode='binary'
)

# MobileNetV2로 특성 추출
mobilenet = MobileNetV2(weights='imagenet', include_top=False, input_shape=(img_height, img_width, 3))

# 레이어 동결
for layer in mobilenet.layers:
    layer.trainable = False

# 특성 추출
train_features = mobilenet.predict(training_set)
test_features = mobilenet.predict(test_set)

# 데이터를 1차원으로 펼침
n_samples, h, w, c = train_features.shape
train_features_flat = train_features.reshape(n_samples, -1)
test_features_flat = test_features.reshape(test_features.shape[0], -1)

# PCA 적용 (32x32로 축소)
n_components = 32 * 32
pca = PCA(n_components=n_components)
train_features_pca = pca.fit_transform(train_features_flat)
test_features_pca = pca.transform(test_features_flat)

# 차원 축소된 데이터 저장
np.save(npyResDir + "train_features_pca.npy", train_features_pca)
np.save(npyResDir + "test_features_pca.npy", test_features_pca)
np.save(npyResDir + "train_labels.npy", training_set.classes)
np.save(npyResDir + "test_labels.npy", test_set.classes)

print("PCA data saved.")

""" Fogging+PCA 시간 측정 종료 """
fog_pca_end_time = time.time()

# 시간 산정
fog_pca_total_time = fog_pca_end_time - fog_pca_start_time
print("\n\n" + "=" * 60)
print("--- Fogging + PCA TimeSet ---")
print(f"Total execute time : {fog_pca_total_time:.3f} seconds.")
print("="*60)
