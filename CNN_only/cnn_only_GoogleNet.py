import os
import time
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import InceptionV3  # GoogleNet (InceptionV3 사용)
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Input
from sklearn.metrics import classification_report

cnn_start_time = time.time()

# 데이터 경로 설정
train_path = "C:\\Users\\Administrator\\Desktop\\ai_fogging_system\\_Dataset\\cat_and_dog\\training_set"
test_path = "C:\\Users\\Administrator\\Desktop\\ai_fogging_system\\_Dataset\\cat_and_dog\\test_set"

# ImageDataGenerator 생성
datagen = ImageDataGenerator(rescale=1.0 / 255.0)

# 훈련 데이터 로드
training_set = datagen.flow_from_directory(
    train_path,
    target_size=(299, 299),  # InceptionV3 입력 크기
    batch_size=32,
    shuffle=True,            # 순서 랜덤화 (학습 시 다양성 추가)
    class_mode='categorical'
)

# 테스트 데이터 로드
test_set = datagen.flow_from_directory(
    test_path,
    target_size=(299, 299),
    batch_size=32,
    shuffle=False,
    class_mode='categorical'
)

# GoogleNet (InceptionV3) 모델 정의
inceptionv3 = InceptionV3(include_top=False, weights='imagenet', input_shape=(299, 299, 3))

# 레이어 동결 (InceptionV3의 가중치 고정)
for layer in inceptionv3.layers:
    layer.trainable = False

# 모델 구성
model = Sequential([
    Input(shape=(299, 299, 3)),
    inceptionv3,                      # InceptionV3 모델 추가
    Flatten(),                        # 특징 맵을 1D 벡터로 변환
    Dense(64, activation='relu'),     # 은닉층
    Dense(2, activation='softmax')    # 출력층 (다중 클래스 분류)
])

model.summary()

# 모델 컴파일
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# 학습 데이터와 검증 데이터 분리
X_train, y_train = next(iter(training_set))
X_test, y_test = next(iter(test_set))

# 모델 학습
history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=10,
    batch_size=32
)

cnn_end_time = time.time()
cnn_execution_time = cnn_end_time - cnn_start_time

# 성능 평가 보고서 함수
def report():
    # 테스트 평가
    test_loss, test_acc = model.evaluate(test_set)

    # Prediction 평가
    test_prediction = model.predict(X_test)
    y_pred = test_prediction.argmax(axis=1)
    y_true = y_test.argmax(axis=1)

    class_names = list(test_set.class_indices.keys())

    # 성능 평가 보고서 생성
    print("\n" + "="*40)
    print("Model Evaluation Report")
    print("="*40)
    print(classification_report(y_true, y_pred, target_names=class_names, labels=range(len(class_names)), zero_division=0))

    # 모델 정확도 출력
    print(f"\nTest Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}")

    # 처리 시간 출력
    print("\n" + "="*40)
    print(f"* CNN model execution time: {cnn_execution_time:.6f} seconds.")
    print("="*40)

report()
