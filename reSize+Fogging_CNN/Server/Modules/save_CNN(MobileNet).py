# MobileNetV2 Training and Save

# 차원 축소된 데이터를 사용한 분류 모델 학습

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical

# 차원 축소 데이터 로드
train_features_reduced = np.load("train_features_reduced.npy")
test_features_reduced = np.load("test_features_reduced.npy")
train_labels = np.load("train_labels.npy")
test_labels = np.load("test_labels.npy")

# 라벨 정규화 (One-hot Encoding)
train_labels_cat = to_categorical(train_labels)
test_labels_cat = to_categorical(test_labels)

# 분류 모델 구성
model = Sequential([
    Input(shape=(train_features_reduced.shape[1],)),  # 차원 축소된 입력
    Dense(128, activation='relu'),  # 은닉층
    Dropout(0.5),
    Dense(64, activation='relu'),  # 추가 은닉층
    Dropout(0.5),
    Dense(train_labels_cat.shape[1], activation='softmax')  # 출력층 (다중 클래스 분류)
])

# 모델 컴파일
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

# 모델 학습
history = model.fit(
    train_features_reduced,
    train_labels_cat,
    validation_data=(test_features_reduced, test_labels_cat),
    epochs=50,
    batch_size=32,
    callbacks=[early_stopping],
    verbose=1
)

# 모델 저장
model.save("classification_model_with_autoencoder.h5")
print("classification model saved.")