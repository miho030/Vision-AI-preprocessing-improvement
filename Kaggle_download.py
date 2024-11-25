import os
import shutil
from kaggle.api.kaggle_api_extended import KaggleApi

# Kaggle 데이터 다운로드
dataset_path = "./_Dataset/cat_dog/"
os.makedirs(dataset_path, exist_ok=True)

api = KaggleApi()
api.authenticate()
api.dataset_download_files("tongpython/cat-and-dog", path=dataset_path, unzip=True)
print("Kaggle dataset download complete!")

# 디렉터리 이동 및 정리
# 1. test_set/cats와 test_set/dogs 이동
src_test_cats = os.path.join(dataset_path, "test_set/test_set/cats")
src_test_dogs = os.path.join(dataset_path, "test_set/test_set/dogs")
dst_test = os.path.join(dataset_path, "test_set")

shutil.move(src_test_cats, dst_test)
shutil.move(src_test_dogs, dst_test)

# 2. training_set/cats와 training_set/dogs 이동
src_train_cats = os.path.join(dataset_path, "training_set/training_set/cats")
src_train_dogs = os.path.join(dataset_path, "training_set/training_set/dogs")
dst_train = os.path.join(dataset_path, "training_set")

shutil.move(src_train_cats, dst_train)
shutil.move(src_train_dogs, dst_train)

# 3. test_set/test_set 디렉터리 삭제
src_test_set_dir = os.path.join(dataset_path, "test_set/test_set")
if os.path.exists(src_test_set_dir):
    shutil.rmtree(src_test_set_dir)

# 4. training_set/training_set 디렉터리 삭제
src_train_set_dir = os.path.join(dataset_path, "training_set/training_set")
if os.path.exists(src_train_set_dir):
    shutil.rmtree(src_train_set_dir)

print("Done!")
