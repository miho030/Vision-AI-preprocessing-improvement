import os, sys, time
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import tensorflow as tf
import numpy as np

# 현재 프로젝트의 경로 활용
cwd = os.path.dirname(os.path.abspath(__file__))

# GPU 설정
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print(f"사용 가능한 GPU {len(gpus)}개: {[gpu.name for gpu in gpus]}")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
else:
    print("사용 가능한 GPU가 없습니다. CPU로 실행합니다.")

def get_dataset_paths():
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
        print("Invalid choice. Please select 1, 2")
        return None, None, None

    print("="*60)
    print("--- DataSet selection ---")
    print(f"Training Path: {train_path}")
    print(f"Testing Path: {test_path}")
    print("="*60)

    return train_path, test_path, choice

# 함수 호출 및 결과 출력
train_path, test_path, choice = get_dataset_paths()
if train_path == None:
    exit(1)

# ImageDataGenerator 생성
datagen = ImageDataGenerator(rescale=1.0 / 255.0)
BATCH_SIZE = 32
img_height, img_width = 224, 224  # InceptionV3 입력 크기

# 시간 측정 시작
cnn_start_time = time.time()

# 훈련 데이터 로드
training_set = datagen.flow_from_directory(
    train_path,
    target_size=(img_height, img_width),
    batch_size=BATCH_SIZE,
    shuffle=True,  # 데이터 섞기
    class_mode='categorical'
)

# 테스트 데이터 로드
test_set = datagen.flow_from_directory(
    test_path,
    target_size=(img_height, img_width),
    batch_size=BATCH_SIZE,
    shuffle=False,
    class_mode='categorical'
)

# InceptionV3 모델 구성
inception = InceptionV3(weights='imagenet', include_top=False, input_shape=(img_height, img_width, 3))

# InceptionV3 레이어 동결 (전이 학습)
for layer in inception.layers:
    layer.trainable = False

# 분류 모델 구성
    # 조건에 따른 모델 구성
    if choice == "1":
        model = Sequential([
            inception,  # InceptionV3 기반 특성 추출
            GlobalAveragePooling2D(),  # 특성 맵을 1D 벡터로 변환
            Dropout(0.5),  # 드롭아웃으로 과적합 방지
            Dense(128, activation='relu'),  # 은닉층
            Dense(64, activation='relu'),  # 추가 은닉층
            Dense(2, activation='sigmoid')  # 출력층
        ])
    elif choice == "2":
        model = Sequential([
            inception,  # InceptionV3 기반 특성 추출
            GlobalAveragePooling2D(),  # 특성 맵을 1D 벡터로 변환
            Dropout(0.5),  # 드롭아웃으로 과적합 방지
            Dense(128, activation='relu'),  # 은닉층
            Dense(64, activation='relu'),  # 추가 은닉층
            Dense(3, activation='sigmoid')  # 출력층
        ])



# 모델 컴파일
model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='categorical_crossentropy',  # 이진 분류 손실 함수
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
    training_set,
    validation_data=test_set,
    epochs=50,
    batch_size=BATCH_SIZE,
    callbacks=[early_stopping],
    verbose=1
)

cnn_end_time = time.time()
cnn_total_time = cnn_end_time - cnn_start_time

# 평가
test_steps = int(np.ceil(test_set.samples / test_set.batch_size))  # 테스트 배치 수 정확히 계산
predictions = model.predict(test_set, steps=test_steps, verbose=1)

# 모델 예측 결과 클래스 변환
predicted_classes = np.argmax(predictions, axis=1)  # 예측 결과를 클래스 레이블로 변환

# 실제 클래스 (test_set.classes 사용)
true_classes = test_set.classes  # 실제 클래스
true_classes = true_classes[:len(predicted_classes)]  # 예측된 클래스 길이에 맞게 조정

# 정확도, 정밀도, 재현율, F1 점수 계산
accuracy = accuracy_score(true_classes, predicted_classes)
precision = precision_score(true_classes, predicted_classes, average='weighted')
recall = recall_score(true_classes, predicted_classes, average='weighted')
f1 = f1_score(true_classes, predicted_classes, average='weighted')

# 결과 출력
print("\n\n" + "=" * 60)
print("--- Model result ---")
print("* CNN / InceptionV3 / Size=None")
print("=" * 60)
print(f"- Accuracy: {accuracy:.4f}")
print(f"- Precision: {precision:.4f}")
print(f"- Recall: {recall:.4f}")
print(f"- F1 Score: {f1:.4f}")
print("=" * 60)
print(f"* CNN model execute time : {cnn_total_time:.3f} seconds.")
print("=" * 60)

"""
# 평가
test_steps = test_set.samples // test_set.batch_size
predictions = model.predict(test_set, steps=test_steps, verbose=1)
predicted_classes = (predictions > 0.5).astype(int).flatten()

# 정확도, 정밀도, 재현율, F1 점수 계산
true_classes = test_set.classes[:len(predicted_classes)]
accuracy = accuracy_score(true_classes, predicted_classes)
precision = precision_score(true_classes, predicted_classes)
recall = recall_score(true_classes, predicted_classes)
f1 = f1_score(true_classes, predicted_classes)

# 결과 출력
print("\n\n" + "=" * 60)
print("--- Model result ---")
print("* CNN / InceptionV3 / Size=None")
print("="*60)
print(f"- Accuracy: {accuracy:.4f}")
print(f"- Precision: {precision:.4f}")
print(f"- Recall: {recall:.4f}")
print(f"- F1 Score: {f1:.4f}")
print("="*60)
print(f"* CNN model execute time : {cnn_total_time:.3f} seconds.")
print("="*60)
"""