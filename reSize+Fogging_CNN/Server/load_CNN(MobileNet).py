# Load classification model

import numpy as np
from tensorflow.keras.models import load_model
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# 차원 축소 데이터 로드
test_features_reduced = np.load("test_features_reduced.npy")
test_labels = np.load("test_labels.npy")

# 저장된 모델 로드
model = load_model("classification_model_with_autoencoder.h5")
print("분류 모델 로드 완료!")

# 데이터 예측
predictions = model.predict(test_features_reduced)
predicted_classes = np.argmax(predictions, axis=1)

# 평가
accuracy = accuracy_score(test_labels, predicted_classes)
precision = precision_score(test_labels, predicted_classes, average='weighted')
recall = recall_score(test_labels, predicted_classes, average='weighted')
f1 = f1_score(test_labels, predicted_classes, average='weighted')

# 결과 출력
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")