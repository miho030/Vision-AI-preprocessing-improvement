# Load classification model
import os, time
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

""" 시간 측정 시작 """
cnn_start_time = time.time()

# 차원 축소 데이터 로드
test_features_reduced = np.load("test_features_reduced.npy")
test_labels = np.load("test_labels.npy")

# 저장된 모델 로드
model = load_model("classification_model_with_autoencoder.h5")
print("분류 모델 로드 완료!")


""" CNN 시간 측정 종료 """
cnn_end_time = time.time()
cnn_total_time = cnn_end_time - cnn_start_time

# 데이터 예측
predictions = model.predict(test_features_reduced)
predicted_classes = np.argmax(predictions, axis=1)

# 평가
accuracy = accuracy_score(test_labels, predicted_classes)
precision = precision_score(test_labels, predicted_classes, average='weighted')
recall = recall_score(test_labels, predicted_classes, average='weighted')
f1 = f1_score(test_labels, predicted_classes, average='weighted')


# 결과 출력
print("\n\n", "=" * 40)
print("\t --- Model result ---")
print("\t * Fogging + CNN Server / MobileNet / Size(32x32)")
print("="*40)
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")
print("="*40)
print(f"CNN model execute time : {cnn_total_time:.3f} seconds.")
print("="*40)