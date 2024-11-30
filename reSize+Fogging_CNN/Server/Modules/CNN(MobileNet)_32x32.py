# CNN(MobileNet)

import os, time
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


# 차원 축소된 데이터 로드
train_features_pca = np.load("train_features_pca.npy")
test_features_pca = np.load("test_features_pca.npy")
train_labels = np.load("train_labels.npy")
test_labels = np.load("test_labels.npy")

# 라벨 정규화 (One-hot Encoding)
train_labels_cat = to_categorical(train_labels)
test_labels_cat = to_categorical(test_labels)

# 분류 모델 구성
model = Sequential([
    Input(shape=(train_features_pca.shape[1],)),  # PCA로 축소된 입력 크기
    Dense(128, activation='relu'),  # 은닉층
    Dropout(0.5),
    Dense(64, activation='relu'),  # 추가 은닉층
    Dropout(0.5),
    Dense(train_labels_cat[1], activation='softmax')  # 출력층 (2 클래스 분류)
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
    train_features_pca,
    train_labels_cat,
    validation_data=(test_features_pca, test_labels_cat),
    epochs=50,
    batch_size=32,
    callbacks=[early_stopping],
    verbose=1
)

# 모델 저장
model_path = os.path.join("", "classification_mobilenet_cnn_dogcat.h5")
model.save(model_path)
print("Model saved.")

# 평가
predictions = model.predict(test_features_pca)
predicted_classes = np.argmax(predictions, axis=1)

# 정확도, 정밀도, 재현율, F1 점수 계산
accuracy = accuracy_score(test_labels, predicted_classes)
precision = precision_score(test_labels, predicted_classes)
recall = recall_score(test_labels, predicted_classes)
f1 = f1_score(test_labels, predicted_classes)

# 결과 출력
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")
