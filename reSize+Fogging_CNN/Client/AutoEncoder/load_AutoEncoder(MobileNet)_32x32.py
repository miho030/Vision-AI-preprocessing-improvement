# Load AutoEncoder

import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2

# 데이터 경로
train_path = "training_set"
test_path = "test_set"

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
encoder = load_model("encoder_32x32_model.h5")
print("Encoder model Loaded.")

# Encoder를 사용하여 차원 축소
train_features_reduced = encoder.predict(train_features)
test_features_reduced = encoder.predict(test_features)

# 차원 축소된 데이터 저장
np.save("train_features_reduced.npy", train_features_reduced)
np.save("test_features_reduced.npy", test_features_reduced)
np.save("train_labels.npy", training_set.classes)
np.save("test_labels.npy", test_set.classes)

print("Autoencoder data saved.")