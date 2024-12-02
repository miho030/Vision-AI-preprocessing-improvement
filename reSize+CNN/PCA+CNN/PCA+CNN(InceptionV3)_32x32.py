# PCA + CNN (InceptionV3) 32x32

import os, sys, time
import numpy as np
from sklearn.decomposition import PCA
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Input, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# 현재 프로젝트의 경로 활용
cwd = os.path.dirname(os.path.abspath(__file__))

# 차원축소 데이터 저장 경로 지정
npyResDir = "./pca_npyRes/pca_cnn_I_32/"
os.makedirs(npyResDir, exist_ok=True)

def get_dataset_paths():
    def get_folder_size(path):
        total_size = 0
        for dirpath, dirnames, filenames in os.walk(path):
            for file in filenames:
                filepath = os.path.join(dirpath, file)
                total_size += os.path.getsize(filepath) # 파일 크기를 누적
        return total_size

    if len(sys.argv) != 2:
        print("\n" + "Usage: python script_name.py <choice> (1, 2, or 3)")
        return None, None, None

    choice = sys.argv[1]

    if choice == "1":
        train_path = os.path.abspath(os.path.join(cwd, "../../_Dataset/cat_dog/training_set"))
        test_path = os.path.abspath(os.path.join(cwd, "../../_Dataset/cat_dog/test_set"))
    elif choice == "2":
        train_path = os.path.abspath(os.path.join(cwd, "../../_Dataset/FER_2013/training_set"))
        test_path = os.path.abspath(os.path.join(cwd, "../../_Dataset/FER_2013/test_set"))
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


"""PCA + 전처리 시간 측정 시작 """
pca_start_time = time.time()

# ImageDataGenerator 생성
datagen = ImageDataGenerator(rescale=1.0 / 255.0)
BATCH_SIZE = 32
img_height, img_width = 224, 224

# 훈련 데이터 로드
training_set = datagen.flow_from_directory(
    train_path,
    target_size=(img_height, img_width),  # 원본 이미지 크기
    batch_size=32,           # 배치 크기 32로 설정
    shuffle=False,           # 순서 고정 (PCA에 중요)
    class_mode='categorical'
)

# 테스트 데이터 로드
test_set = datagen.flow_from_directory(
    test_path,
    target_size=(img_height, img_width),
    batch_size=32,           # 배치 크기 32로 설정
    shuffle=False,
    class_mode='categorical'
)

# 전이학습 모델 불러오기
inception = InceptionV3(weights='imagenet', include_top=False, input_shape=(img_height, img_width, 3))

# 레이어 동결
for layer in inception.layers:
    layer.trainable = False

# 전이학습 모델로 특성 추출
train_features = inception.predict(training_set)
test_features = inception.predict(test_set)

# 데이터를 1차원으로 펼침
n_samples, h, w, c = train_features.shape
train_features_flat = train_features.reshape(n_samples, -1)
test_features_flat = test_features.reshape(test_features.shape[0], -1)

# PCA로 차원 축소 대상 차원 설정
n_components = 32*32
pca = PCA(n_components=n_components)
train_features_pca = pca.fit_transform(train_features_flat)
test_features_pca = pca.transform(test_features_flat)

# PCA 이후 데이터 정규화
train_labels = training_set.classes
test_labels = test_set.classes
train_labels_cat = to_categorical(train_labels)
test_labels_cat = to_categorical(test_labels)


""" PCA + 전처리 종료 시점, 시간 측정 종료"""
pca_end_time = time.time()


""" CNN 모델 세팅, 학습 시간 측정 시작"""
cnn_start_time = time.time()

# 모델 구성

if choice == "1":
    output = Dense(2, activation='softmax')  # 출력층
elif choice == "2":
    output = Dense(5, activation='softmax')  # 출력층
elif choice == "3":
    output = Dense(3, activation='softmax')  # 출력층

model = Sequential([
    Input(shape=(n_components,)),
    Dense(128, activation='relu'),  # 은닉층
    Dropout(0.5),
    Dense(64, activation='relu'),  # 추가 은닉층
    Dropout(0.5),
    output
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
    patience=5,  # 성능 개선이 없을 때 학습 중지
    restore_best_weights=True
)

# 모델 학습
history = model.fit(
    train_features_pca,
    train_labels_cat,
    validation_data=(test_features_pca, test_labels_cat),
    epochs=50,
    batch_size=BATCH_SIZE,
    callbacks=[early_stopping],
    verbose=1
)

""" CNN 세팅, 학습 시간 측정 종료 시점 """
cnn_end_time = time.time()

# 시간 산정
pca_total_time = pca_end_time - pca_start_time
cnn_total_time = cnn_end_time - cnn_start_time
total_time = cnn_total_time + pca_total_time

# 평가
predictions = model.predict(test_features_pca)
predicted_classes = np.argmax(predictions, axis=1)

# 정확도, 정밀도, 재현율, F1 점수 계산
accuracy = accuracy_score(test_labels, predicted_classes)
precision = precision_score(test_labels, predicted_classes, average='micro')
recall = recall_score(test_labels, predicted_classes, average='micro')
f1 = f1_score(test_labels, predicted_classes, average='micro')


# 학습된 가중치 모델 저장
if choice == "1":
    model_path = os.path.join("", "classification_inceptiont_32x32_dogcat.h5")
    model.save(model_path)
elif choice == "3":
    model_path = os.path.join("", "classification_inceptiont_32x32_FEB2013.h5")
    model.save(model_path)
print("h5 Model saved.")


# 차원 축소된 데이터 저장 및 용량 확인
def measure_npy_file_sizes(directory):
    print("\n\n" + "=" * 60)
    np.save(npyResDir + "train_features_pca.npy", train_features_pca)
    np.save(npyResDir + "test_features_pca.npy", test_features_pca)
    np.save(npyResDir + "train_labels.npy", training_set.classes)
    np.save(npyResDir + "test_labels.npy", test_set.classes)
    print(f" -> PCA data saved at {npyResDir}..\n" + "="*60)
    print(f"--- File sizes in {directory} ---\n")

    total_size = 0
    # 디렉토리의 모든 파일 확인
    for file_name in os.listdir(directory):
        file_path = os.path.join(directory, file_name)
        if os.path.isfile(file_path):  # 파일인지 확인
            file_size = os.path.getsize(file_path)  # 파일 크기 (바이트)
            print(f"- {file_name}: {file_size / (1024 ** 2):.2f} MB")  # MB 단위로 출력
            total_size += file_size

    # 총 용량 출력
    print(f"\nTotal size of all files: {total_size / (1024 ** 2):.2f} MB\n" + "="*60)
# npyResDir 내 파일 용량 측정 함수 호출
measure_npy_file_sizes(npyResDir)

# 결과 출력
print("\n\n" + "=" * 60)
print("--- Model result ---")
print("* PCA + CNN / InceptionV3 / Size(32x32)")
print("="*60)
print(f"- Accuracy: {accuracy:.4f}")
print(f"- Precision: {precision:.4f}")
print(f"- Recall: {recall:.4f}")
print(f"- F1 Score: {f1:.4f}")
print("="*60)
print(f"* PCA execute time : {pca_total_time:.3f} seconds.")
print(f"* CNN model execute time : {cnn_total_time:.3f} seconds.")
print(f"* Total execute time : {total_time:.3f} seconds.")
print("="*60)