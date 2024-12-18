import os
import subprocess

#from.calculate_folder_sizes import calculate_folder_and_subfolder_sizes

def execute_scripts(script_paths, input_values, log_dir="logs"):
    """
    스크립트를 실행하고, input 값을 전달한 뒤, 로그를 저장합니다.

    :param script_paths: 실행할 스크립트 파일 경로 리스트
    :param input_values: 각 스크립트에 전달할 입력값 리스트
    :param log_dir: 로그 파일을 저장할 디렉터리
    """
    # 로그 디렉터리가 없으면 생성
    os.makedirs(log_dir, exist_ok=True)

    for script in script_paths:
        for input_value in input_values:
            log_filename = f"{input_value}_{os.path.basename(script).replace('.py', '')}.txt"
            log_filepath = os.path.join(log_dir, log_filename)

            try:
                with open(log_filepath, "w") as log_file:
                    # subprocess로 스크립트 실행 (입력값을 명령줄 인수로 전달)
                    process = subprocess.run(
                        ["python3", script, str(input_value)],  # 'python script_path input_value'
                        text=True,
                        stdout=log_file,  # 표준 출력을 로그 파일로 저장
                        stderr=log_file,  # 표준 에러도 로그 파일로 저장
                    )
                    print(f"Executed {script} with input {input_value}, log saved to {log_filepath}")
            except Exception as e:
                print(f"Failed to execute {script} with input {input_value}: {e}")


# 스크립트 경로 설정
base_folder = "."  # 현재 폴더를 기준으로 설정

script_paths = []

# AutoEncoder 폴더에서 스크립트 수집
autoencoder_folder = os.path.join(base_folder, "AutoEncoder")
for root, dirs, files in os.walk(autoencoder_folder):
    for file in files:
        if file.endswith(".py"):
            script_paths.append(os.path.join(root, file))

# PCA 폴더에서 스크립트 수집
pca_folder = os.path.join(base_folder, "PCA")
for root, dirs, files in os.walk(pca_folder):
    for file in files:
        if file.endswith(".py"):
            script_paths.append(os.path.join(root, file))

# save_로 시작하는 파일을 우선적으로 실행하도록 정렬
script_paths.sort(key=lambda x: (not os.path.basename(x).startswith("save_"), x))

# 테스트 코드: script_paths 내용 출력
print("=== Collected Script Paths ===")
for script in script_paths:
    print(script)
print("================================")

# 입력값 정의
input_values = [1, 2]
# 주요 디렉토리 지정
base_directories = ["./_Dataset", "./pca_npyRes", "./auto_npyRes"]

# 스크립트 실행 및 로그 저장
execute_scripts(script_paths, input_values)
# 스크립트 실행 이후 파일 크기 계산
# calculate_folder_and_subfolder_sizes(base_directories)
