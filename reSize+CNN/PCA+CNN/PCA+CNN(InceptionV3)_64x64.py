# PCA + CNN (InceptionV3) 64x64
# -*- coding: utf-8 -*-

import os, time
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Input
from sklearn.model_selection import train_test_split
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

train_path = "../../_Dataset/cat_dog/training_set"
test_path = "../../_Dataset/cat_dog/test_set"

""" PCA, 전처리 시간 측정 시작 """
pca_start_time = time.time()

datagen = ImageDataGenerator(rescale=1.0 / 255.0)
BATCH_SIZE = 32
img_height, img_width = 224, 224

training_set = datagen.flow_from_directory(
    train_path,
    target_size=(img_height, img_width),  # ���� �̹��� ũ��
    batch_size=32,           # ��ġ ũ�� 32�� ����
    shuffle=False,           # ���� ���� (PCA�� �߿�)
    class_mode='binary'
)

test_set = datagen.flow_from_directory(
    test_path,
    target_size=(img_height, img_width),
    batch_size=32,           # ��ġ ũ�� 32�� ����
    shuffle=False,
    class_mode='binary'
)

inception = InceptionV3(weights='imagenet', include_top=False, input_shape=(img_height, img_width, 3))

for layer in inception.layers:
    layer.trainable = False

train_features = inception.predict(training_set)
test_features = inception.predict(test_set)

n_samples, h, w, c = train_features.shape
train_features_flat = train_features.reshape(n_samples, -1)
test_features_flat = test_features.reshape(test_features.shape[0], -1)

n_components = 64*64
pca = PCA(n_components=n_components)
train_features_pca = pca.fit_transform(train_features_flat)
test_features_pca = pca.transform(test_features_flat)

train_labels = training_set.classes
test_labels = test_set.classes
train_labels_cat = to_categorical(train_labels)
test_labels_cat = to_categorical(test_labels)

""" PCA, 전처리 시간 측정 종료 """
pca_end_time = time.time()


""" CNN 시간 측정 시작 """
cnn_start_time = time.time()

model = Sequential([
    Input(shape=(n_components,)),
    Dense(128, activation='relu'),  # ������
    Dense(64, activation='relu'),  # �߰� ������
    Dense(2, activation='softmax')  # ����� (2 Ŭ���� �з�)
])

model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=5,  # ���� ������ ���� �� �н� ����
    restore_best_weights=True
)

history = model.fit(
    train_features_pca,
    train_labels_cat,
    validation_data=(test_features_pca, test_labels_cat),
    epochs=50,
    batch_size=BATCH_SIZE,
    callbacks=[early_stopping],
    verbose=1
)

model_path = os.path.join("", "classification_inceptiont_64x64_dogcat.h5")
model.save(model_path)
print("Model saved.")

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
precision = precision_score(test_labels, predicted_classes)
recall = recall_score(test_labels, predicted_classes)
f1 = f1_score(test_labels, predicted_classes)


# 결과 출력
print("\n\n", "=" * 40)
print("\t --- Model result ---")
print("\t * PCA + CNN / InceptionV3 / Size(64x64)")
print("="*40)
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")
print("="*40)
print(f"PCA execute time : {pca_total_time:.6f} seconds.")
print(f"CNN model execute time : {cnn_total_time:.6f} seconds.")
print(f"Total execute time : {total_time:.6f} seconds.")
print("="*40)