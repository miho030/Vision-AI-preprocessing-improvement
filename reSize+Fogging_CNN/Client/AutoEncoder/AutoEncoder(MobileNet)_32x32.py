# AutoEncoder (MobileNetV2)

import os, sys, time
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, Flatten, Reshape, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

# 현재 프로젝트의 경로 활용
cwd = os.path.dirname(os.path.abspath(__file__))

# 차원축소 데이터 저장 경로 지정
npyResDir = "./auto_npyRes/fog_auto_M_32/"
os.makedirs(npyResDir, exist_ok=True)

# 재귀함수 활용하여 결정된 데이터셋의 절대경로 및 용량을 구해 출력함.
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
        train_path = os.path.abspath(os.path.join(cwd, "../_Dataset/cat_dog/training_set"))
        test_path = os.path.abspath(os.path.join(cwd, "../_Dataset/cat_dog/test_set"))
    elif choice == "2":
        train_path = os.path.abspath(os.path.join(cwd, "../_Dataset/FER_2013/training_set"))
        test_path = os.path.abspath(os.path.join(cwd, "../_Dataset/FER_2013/test_set"))
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
# 함수 호출 및 결과 출력
train_path, test_path, choice = get_dataset_paths()
if train_path == None:
    exit(1)

""" 시간 측정 시작 """
auto_start_time = time.time()

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
np.save(npyResDir + "train_features_reduced.npy", train_features_reduced)
np.save(npyResDir + "test_features_reduced.npy", test_features_reduced)
np.save(npyResDir + "train_labels.npy", training_set.classes)
np.save(npyResDir + "test_labels.npy", test_set.classes)

print("Autoencoder data saved.")

""" 시간 측정 종료 """
auto_end_time = time.time()
auto_total_time = auto_end_time - auto_start_time

print("\n\n", "=" * 40)
print("--- Fogging + (load)AutoEncoder (MobileNet) | Size(32x32) TimeSet ---")
print(f"* Total execute time : {auto_total_time:.3f} seconds.")
print("="*40)

