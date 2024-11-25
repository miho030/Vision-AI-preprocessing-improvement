# AutoEncoder training and save

import os, time
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten, Reshape, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2

# 데이터 경로
train_path = os.path.abspath("../../../_Dataset/cat_dog/training_set")
test_path = os.path.abspath("../../../_Dataset/cat_dog/test_set")
npyResDir = "./auto_npyRes/"

os.makedirs(npyResDir, exist_ok=True)

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

# MobileNetV2를 이용한 특성 추출
mobilenet = MobileNetV2(weights='imagenet', include_top=False, input_shape=(img_height, img_width, 3))
for layer in mobilenet.layers:
    layer.trainable = False

train_features = mobilenet.predict(training_set)
test_features = mobilenet.predict(test_set)

# AutoEncoder 모델 구성
encoded_dim = (32, 32)
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

# 모델 저장
h5_file_path = npyResDir + "encoder_32x32_model.h5"
encoder.save(h5_file_path)
print("Encoder model saved.")

""" 시간 측정 종료 """
auto_end_time = time.time()
auto_total_time = auto_end_time - auto_start_time

# 시간 산정
print("\n\n" + "=" * 60)
print("--- Fogging + AutoEncoder (save) TimeSet ---")
print(f" * Total execute time : {auto_total_time:.3f} seconds.")
print("="*60)