# AutoEncoder (MobileNetV2)

import os
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, Flatten, Reshape, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

train_path = "training_set"
test_path = "test_set"

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

# AutoEncoder 모델 정의
encoded_dim = (32, 32)  # AutoEncoder의 압축 차원
input_dim = train_features.shape[1:]  # MobileNetV2의 출력 크기

# Encoder
encoder_input = Input(shape=input_dim)
x = Flatten()(encoder_input)
encoded = Dense(np.prod(encoded_dim), activation='relu')(x)
encoded_reshaped = Reshape(encoded_dim)(encoded)

# Decoder
decoder_flat = Flatten()(encoded_reshaped)
decoder_output = Dense(np.prod(input_dim), activation='relu')(decoder_flat)
decoder_output_reshaped = Reshape(input_dim)(decoder_output)

# AutoEncoder 모델
autoencoder = Model(encoder_input, decoder_output_reshaped)
autoencoder.compile(optimizer=Adam(learning_rate=0.001), loss='mse')

# Encoder 모델 분리
encoder = Model(encoder_input, encoded_reshaped)

# AutoEncoder 학습
autoencoder.fit(
    train_features, train_features,
    validation_data=(test_features, test_features),
    epochs=50,
    batch_size=BATCH_SIZE,
    verbose=1,
    callbacks=[EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)]
)

# 차원 축소된 데이터 생성
train_features_reduced = encoder.predict(train_features)
test_features_reduced = encoder.predict(test_features)

# 차원 축소된 데이터 저장
np.save("train_features_reduced.npy", train_features_reduced)
np.save("test_features_reduced.npy", test_features_reduced)
np.save("train_labels.npy", training_set.classes)
np.save("test_labels.npy", test_set.classes)

print("Autoencoder data saved.")