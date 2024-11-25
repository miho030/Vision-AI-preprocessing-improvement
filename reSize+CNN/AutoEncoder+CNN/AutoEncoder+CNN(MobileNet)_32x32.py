# Autoencoder + CNN (MobileNetV2) 32x32

import os, time
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input, Flatten, Reshape
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


train_path = "../../_Dataset/cat_dog/training_set"
test_path = "../../_Dataset/cat_dog/test_set"


""" AutoEncoder 시간 측정 시작 """
auto_start_time = time.time()

# ImageDataGenerator 생성
datagen = ImageDataGenerator(rescale=1.0 / 255.0)
BATCH_SIZE = 32
img_height, img_width = 224, 224  # MobileNet 입력 크기

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

# 전이학습 모델 불러오기 (MobileNetV2)
mobilenet = MobileNetV2(weights='imagenet', include_top=False, input_shape=(img_height, img_width, 3))

# 레이어 동결
for layer in mobilenet.layers:
    layer.trainable = False

# MobileNetV2로 특성 추출
train_features = mobilenet.predict(training_set)
test_features = mobilenet.predict(test_set)

# Autoencoder 모델 구성
encoded_dim = (32, 32)  # Autoencoder의 압축 크기
input_dim = train_features.shape[1:]

# Encoder
encoder_input = Input(shape=input_dim)
x = Flatten()(encoder_input)
encoded = Dense(np.prod(encoded_dim), activation='relu')(x)
encoded_reshaped = Reshape(encoded_dim)(encoded)

# Decoder
decoder_flat = Flatten()(encoded_reshaped)
decoder_output = Dense(np.prod(input_dim), activation='relu')(decoder_flat)
decoder_output_reshaped = Reshape(input_dim)(decoder_output)

# Autoencoder 모델
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

# Autoencoder로 차원 축소
train_features_reduced = encoder.predict(train_features)
test_features_reduced = encoder.predict(test_features)

# Flatten for classification input
train_features_flat = train_features_reduced.reshape(train_features_reduced.shape[0], -1)
test_features_flat = test_features_reduced.reshape(test_features_reduced.shape[0], -1)

# 라벨 정규화
train_labels = training_set.classes
test_labels = test_set.classes
train_labels_cat = to_categorical(train_labels)
test_labels_cat = to_categorical(test_labels)

""" AutoEncoder 시간 측정 시작 """
auto_end_time = time.time()


""" CNN 시간 측정 시작 """
cnn_start_time = time.time()

# 분류 모델 구성
model = Sequential([
    Input(shape=(np.prod(encoded_dim),)),
    Dense(128, activation='relu'),  # 은닉층
    Dense(64, activation='relu'),  # 추가 은닉층
    Dense(2, activation='softmax')  # 출력층 (2 클래스 분류)
])

# 분류 모델 컴파일
model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# EarlyStopping 설정
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)

# 분류 모델 학습
history = model.fit(
    train_features_flat,
    train_labels_cat,
    validation_data=(test_features_flat, test_labels_cat),
    epochs=50,
    batch_size=BATCH_SIZE,
    callbacks=[early_stopping],
    verbose=1
)

# AutoEncoder 저장
ae_path = os.path.join("", "autoencoder_mobilenet_32x32_dogcat.h5")
autoencoder.save(ae_path)
print("AutoEncoder saved.")

# 분류 모델 저장
model_path = os.path.join("", "Classification_mobilenet_32x32_dogcat.h5")
model.save(model_path)
print("Model saved.")

""" CNN 모델 세팅, 학습 시간 측정 종료 시점 """
cnn_end_time = time.time()

# 시간 산정
autoEncoder_total_time = auto_end_time - auto_end_time
cnn_total_time = cnn_end_time - cnn_start_time
total_time = autoEncoder_total_time + cnn_total_time

# 평가
predictions = model.predict(test_features_flat)
predicted_classes = np.argmax(predictions, axis=1)

# 정확도, 정밀도, 재현율, F1 점수 계산
accuracy = accuracy_score(test_labels, predicted_classes)
precision = precision_score(test_labels, predicted_classes)
recall = recall_score(test_labels, predicted_classes)
f1 = f1_score(test_labels, predicted_classes)

# 결과 출력
print("\n\n", "=" * 40)
print("\t --- Model result ---")
print("\t * AutoEncoder + CNN / MobileNet / Size(32x32)")
print("="*40)
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")
print("="*40)
print(f"AutoEncoder execute time : {autoEncoder_total_time:.6f} seconds.")
print(f"CNN model execute time : {cnn_total_time:.6f} seconds.")
print(f"Total execute time : {total_time:.6f} seconds.")
print("="*40)
