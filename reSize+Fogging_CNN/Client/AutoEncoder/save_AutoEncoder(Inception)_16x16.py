# AutoEncoder training and save

import os, sys, time
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten, Reshape, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import InceptionV3

# 데이터 경로
cwd = os.path.dirname(os.path.abspath(__file__))

# 차원축소 데이터 저장 경로 지정
npyResDir = "./auto_npyRes/fog_auto_I_16/"
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


""" 시간 측정 시작 """
auto_start_time = time.time()

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

# InceptionV3를 이용한 특성 추출
inception = InceptionV3(weights='imagenet', include_top=False, input_shape=(img_height, img_width, 3))
for layer in inception.layers:
    layer.trainable = False

train_features = inception.predict(training_set)
test_features = inception.predict(test_set)

# AutoEncoder 모델 구성
encoded_dim = (16, 16)
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

# AutoEncoder
autoencoder = Model(encoder_input, decoder_output_reshaped)
autoencoder.compile(optimizer=Adam(learning_rate=0.001), loss='mse')

# Encoder 분리
encoder = Model(encoder_input, encoded_reshaped)

# AutoEncoder 학습
autoencoder.fit(
    train_features, train_features,
    validation_data=(test_features, test_features),
    epochs=50,
    batch_size=BATCH_SIZE,
    callbacks=[EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)],
    verbose=1
)

# AutoEncoder 및 분류 모델 저장
ae_path, model_path = None, None
if choice == "1":
    ae_path = os.path.join(npyResDir, "fog_autoencoder_mobilenet_16x16_dogcat.h5")
elif choice == "3":
    ae_path = os.path.join(npyResDir, "fog_autoencoder_mobilenet_16x16_FEB2013.h5")

if ae_path == None or model_path == None:
    print("autoEncoder and model path is nor correct!!")
    exit(1)
else:
    encoder.save(ae_path)
    print("AutoEncoder saved.")

""" 시간 측정 종료 """
auto_end_time = time.time()
auto_total_time = auto_end_time - auto_start_time


def size_npy_files(directory):
    print("=" * 60)
    total_size = 0
    # 디렉토리의 모든 파일 확인
    for file_name in os.listdir(directory):
        file_path = os.path.join(directory, file_name)
        if os.path.isfile(file_path):  # 파일인지 확인
            file_size = os.path.getsize(file_path)  # 파일 크기 (바이트)
            print(f" - {file_name}: {file_size / (1024 ** 2):.2f} MB")  # MB 단위로 출력
            total_size += file_size

    # 총 용량 출력
    print(f"\nTotal size of all files: {total_size / (1024 ** 2):.2f} MB\n" + "="*60)


# npyResDir 내 파일 용량 측정 함수 호출
size_npy_files(npyResDir)

# 시간 산정
print("\n\n" + "=" * 60)
print("--- Fogging + (save)AutoEncoder (InceptionV3) | Size(16x16) TimeSet ---")
print(f" * Total execute time : {auto_total_time:.3f} seconds.")
print("="*60)