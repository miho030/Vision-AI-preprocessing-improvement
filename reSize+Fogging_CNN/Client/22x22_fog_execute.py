import os
import subprocess

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

# AutoEncoder 폴더에서 '22x22' 문자열이 포함된 스크립트만 수집
autoencoder_folder = os.path.join(base_folder, "AutoEncoder")
for root, dirs, files in os.walk(autoencoder_folder):
    for file in files:
        if file.endswith(".py") and "22x22" in file:
            script_paths.append(os.path.join(root, file))

# PCA 폴더에서 '22x22' 문자열이 포함된 스크립트만 수집
pca_folder = os.path.join(base_folder, "PCA")
for root, dirs, files in os.walk(pca_folder):
    for file in files:
        if file.endswith(".py") and "22x22" in file:
            script_paths.append(os.path.join(root, file))

# 입력값 정의
input_values = [1, 2]

# 스크립트 정렬: save_ → load_ → 기타
save_scripts = [script for script in script_paths if os.path.basename(script).startswith("save_")]
load_scripts = [script for script in script_paths if os.path.basename(script).startswith("load_")]
other_scripts = [script for script in script_paths if not (os.path.basename(script).startswith("save_") or os.path.basename(script).startswith("load_"))]

# 정렬된 순서로 실행
sorted_script_paths = save_scripts + load_scripts + other_scripts

# 주요 디렉토리 지정
base_directories = ["./_Dataset", "./pca_npyRes", "./auto_npyRes"]

# 스크립트 실행 및 로그 저장
execute_scripts(sorted_script_paths, input_values)
