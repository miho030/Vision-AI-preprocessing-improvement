import os, sys, time
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input, Flatten, Reshape, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import tensorflow as tf

class MemoryPrintingCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
      gpu_dict = tf.config.experimental.get_memory_info('GPU:0')
      tf.print('\n GPU memory details [current: {} gb, peak: {} gb]'.format(
          float(gpu_dict['current']) / (1024 ** 3),
          float(gpu_dict['peak']) / (1024 ** 3)))

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)


# 현재 프로젝트의 경로 활용
cwd = os.path.dirname(os.path.abspath(__file__))

# 차원축소 데이터 저장 경로 지정
npyResDir = "./auto_npyRes/auto_cnn_I_64/"
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
        train_path = os.path.abspath(os.path.join(cwd, "../../_Dataset/cat_dog/training_set"))
        test_path = os.path.abspath(os.path.join(cwd, "../../_Dataset/cat_dog/test_set"))
    elif choice == "2":
        train_path = os.path.abspath(os.path.join(cwd, "../../_Dataset/swimcat/training_set"))
        test_path = os.path.abspath(os.path.join(cwd, "../../_Dataset/swimcat/test_set"))
    elif choice == "3":
        train_path = os.path.abspath(os.path.join(cwd, "../../_Dataset/FER_2013/training_set"))
        test_path = os.path.abspath(os.path.join(cwd, "../../_Dataset/FER_2013/test_set"))
    else:
        print("Invalid choice. Please select 1, 2, or 3.")
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

""" AutoEncoder 시간 측정 시작 """
autoEncoder_start_time = time.time()

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
    class_mode='categorical'  # 다중 클래스 분류
)

# 테스트 데이터 로드
test_set = datagen.flow_from_directory(
    test_path,
    target_size=(img_height, img_width),
    batch_size=BATCH_SIZE,
    shuffle=False,
    class_mode='categorical'  # 다중 클래스 분류
)

# 전이학습 모델 불러오기 (InceptionV3)
inception = InceptionV3(weights='imagenet', include_top=False, input_shape=(img_height, img_width, 3))

# 레이어 동결
for layer in inception.layers:
    layer.trainable = False

# InceptionV3로 특성 추출
train_features = inception.predict(training_set)
test_features = inception.predict(test_set)

train_features = train_features.astype("float16")
test_features = test_features.astype("float16")

# Autoencoder 모델 구성
encoded_dim = (64, 64)  # Autoencoder의 압축 크기
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

# Autoencoder 학습
autoencoder.fit(
    train_features, train_features,
    validation_data=(test_features, test_features),
    epochs=50,
    batch_size=BATCH_SIZE,
    verbose=1,
    callbacks=[EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True), MemoryPrintingCallback()]
)

# Autoencoder로 차원 축소
train_features_reduced = encoder.predict(train_features)
test_features_reduced = encoder.predict(test_features)

# 1차원 벡터로 변환
train_features_flat = train_features_reduced.reshape(train_features_reduced.shape[0], -1)
test_features_flat = test_features_reduced.reshape(test_features_reduced.shape[0], -1)

# 라벨 정규화
train_labels = training_set.classes
test_labels = test_set.classes
train_labels_cat = to_categorical(train_labels)
test_labels_cat = to_categorical(test_labels)

""" AutoEncoder + 전처리 시간 측정 종료 시점 """
autoEncoder_end_time = time.time()


""" CNN 세팅, 학습 시간 측정 시작 시점"""
cnn_start_time = time.time()

# 분류 모델 구성

if choice == "1":
    output = Dense(2, activation='softmax')  # 출력층
elif choice == "2":
    output = Dense(5, activation='softmax')  # 출력층
elif choice == "3":
    output = Dense(3, activation='softmax')  # 출력층


model = Sequential([
    Input(shape=(np.prod(encoded_dim),)),
    Dense(128, activation='relu'),  # 은닉층
    Dropout(0.5),
    Dense(64, activation='relu'),  # 추가 은닉층
    Dropout(0.5),
    output
])

# 분류 모델 컴파일
model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='categorical_crossentropy',  # 다중 클래스 분류 손실 함수
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

""" CNN 모델 세팅, 학습 시간 측정 종료 시점 """
cnn_end_time = time.time()

# 시간 산정
autoEncoder_total_time = autoEncoder_end_time - autoEncoder_start_time
cnn_total_time = cnn_end_time - cnn_start_time
total_time = autoEncoder_total_time + cnn_total_time

# 평가
predictions = model.predict(test_features_flat)
predicted_classes = np.argmax(predictions, axis=1)

# 정확도, 정밀도, 재현율, F1 점수 계산
accuracy = accuracy_score(test_labels, predicted_classes)
precision = precision_score(test_labels, predicted_classes, average='weighted')
recall = recall_score(test_labels, predicted_classes, average='weighted')
f1 = f1_score(test_labels, predicted_classes, average='weighted')



# AutoEncoder 및 분류 모델 저장
ae_path, model_path = None, None
if choice == "1":
    ae_path = os.path.join(npyResDir, "autoencoder_inception_64x64_dogcat.h5")
    model_path = os.path.join(npyResDir, "Classification_inception_64x64_dogcat.h5")
elif choice == "3":
    ae_path = os.path.join(npyResDir, "autoencoder_inception_64x64_FEB2013.h5")
    model_path = os.path.join(npyResDir, "Classification_inception_64x64_FEB2013.h5")

if ae_path == None or model_path == None:
    print("autoEncoder and model path is nor correct!!")
    exit(1)
else:
    autoencoder.save(ae_path)
    print("AutoEncoder saved.")
    model.save(model_path)
    print("Model saved.")

# Encoder를 사용하여 차원 축소
train_features_reduced = encoder.predict(train_features)
test_features_reduced = encoder.predict(test_features)

# 차원 축소된 데이터 저장 및 용량 확인
def measure_npy_file_sizes(directory):
    print("\n\n" + "=" * 60)
    print(f"--- reSized demention files ---")
    np.save(npyResDir + "train_features_auto.npy", train_features_reduced)
    np.save(npyResDir + "test_features_auto.npy", test_features_reduced)
    np.save(npyResDir + "train_labels.npy", training_set.classes)
    np.save(npyResDir + "test_labels.npy", test_set.classes)
    print(f"* Autoencoder data saved at {npyResDir}..\n" + "="*60)
    print(f"--- File sizes in {directory} ---\n")

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
measure_npy_file_sizes(npyResDir)

# 결과 출력
print("\n\n" + "=" * 60)
print("--- Model result ---")
print("* AutoEncoder + CNN / InceptionV3 / Size(64x64)")
print("="*60)
print(f"- Accuracy: {accuracy:.4f}")
print(f"- Precision: {precision:.4f}")
print(f"- Recall: {recall:.4f}")
print(f"- F1 Score: {f1:.4f}")
print("="*60)
print(f"* AutoEncoder execute time : {autoEncoder_total_time:.3f} seconds.")
print(f"* CNN model execute time : {cnn_total_time:.3f} seconds.")
print(f"* Total execute time : {total_time:.3f} seconds.")
print("="*60)