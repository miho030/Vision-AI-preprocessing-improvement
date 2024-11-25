import os
import shutil
from sklearn.model_selection import train_test_split

# 데이터 경로 설정
train_dir = r"C:\Users\aoi\Desktop\ai_fogging_system\_Dataset\cat_dog\training_set"
test_dir = r"C:\Users\aoi\Desktop\ai_fogging_system\_Dataset\cat_dog\test_set"

# 출력 디렉토리 설정
output_base_dir = r"C:\Users\aoi\Desktop\ai_fogging_system\fogging_detasets"  # 분배된 데이터 저장 경로

# Fogging 장비 개수
num_devices = 3

# 출력 디렉토리 초기화
if os.path.exists(output_base_dir):
    shutil.rmtree(output_base_dir)
os.makedirs(output_base_dir)


# 하위 디렉토리까지 탐색하여 모든 파일 목록 가져오기
def get_all_files(source_dir):
    all_files = []
    for root, dirs, files in os.walk(source_dir):
        for file in files:
            file_path = os.path.join(root, file)
            all_files.append(file_path)
    return all_files


# 데이터를 3등분하고 저장하는 함수
def distribute_files_sklearn(source_dir, output_dir, num_devices):
    files = get_all_files(source_dir)  # 모든 파일 목록 가져오기

    # 데이터 확인
    if len(files) == 0:
        print(f"No files found in {source_dir}")
        return

    # 첫 번째 분할: 전체 데이터를 1/3 vs 2/3으로 나눔
    device1_files, temp_files = train_test_split(files, test_size=(num_devices - 1) / num_devices, random_state=42)

    # 두 번째 분할: 남은 데이터를 1/2 vs 1/2으로 나눔
    device2_files, device3_files = train_test_split(temp_files, test_size=0.5, random_state=42)

    # 분배된 데이터 리스트
    distributed_files = [device1_files, device2_files, device3_files]

    # 각 Fogging 장비 디렉토리에 파일 저장
    for device_id, device_files in enumerate(distributed_files, start=1):
        device_dir = os.path.join(output_dir, f"device_{device_id}")
        os.makedirs(device_dir, exist_ok=True)

        for file_path in device_files:
            # 상대 경로 계산
            relative_path = os.path.relpath(file_path, source_dir)
            dest_path = os.path.join(device_dir, relative_path)

            # 대상 디렉토리 생성 및 파일 복사
            os.makedirs(os.path.dirname(dest_path), exist_ok=True)
            shutil.copy(file_path, dest_path)

        print(f"Device {device_id}: {len(device_files)} files distributed.")

# 테스트 데이터와 학습 데이터 분배
print("Distributing test set...")
distribute_files_sklearn(test_dir, os.path.join(output_base_dir, "test_set"), num_devices)

print("Distributing training set...")
distribute_files_sklearn(train_dir, os.path.join(output_base_dir, "training_set"), num_devices)

print("데이터 분배가 완료되었습니다.")
