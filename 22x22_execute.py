import os
import subprocess
import gc
from numba import cuda

# GPU 메모리 해제 함수
def clear_gpu_memory():
    print("Clearing GPU memory...")
    cuda.select_device(0)
    cuda.close()
    gc.collect()

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
                        ["python", script, str(input_value)],  # 'python script_path input_value'
                        text=True,
                        stdout=log_file,  # 표준 출력을 로그 파일로 저장
                        stderr=log_file,  # 표준 에러도 로그 파일로 저장
                    )
                    print(f"Executed {script} with input {input_value}, log saved to {log_filepath}")
                    clear_gpu_memory()
            except Exception as e:
                print(f"Failed to execute {script} with input {input_value}: {e}")


# 스크립트 경로 정의
base_folder = "./"  # 최상위 루트 폴더 경로 설정
script_paths = [
    os.path.join(base_folder, "reSize+CNN", "AutoEncoder+CNN", "AutoEncoder+CNN(InceptionV3)_22x22.py"),
    os.path.join(base_folder, "reSize+CNN", "AutoEncoder+CNN", "AutoEncoder+CNN(MobileNet)_22x22.py"),
    os.path.join(base_folder, "reSize+CNN", "PCA+CNN", "PCA+CNN(InceptionV3)_22x22.py"),
    os.path.join(base_folder, "reSize+CNN", "PCA+CNN", "PCA+CNN(MobileNet)_22x22.py"),
]

# 입력값 정의
input_values = [1, 2]

# 스크립트 실행 및 로그 저장
execute_scripts(script_paths, input_values)