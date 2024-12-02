import os


def calculate_folder_and_subfolder_sizes(base_dirs):
    """
    주어진 디렉토리와 하위 폴더의 크기를 계산하여 출력합니다.

    :param base_dirs: 디렉토리 경로 리스트
    """
    for base_dir in base_dirs:
        if not os.path.exists(base_dir):
            print(f"경로가 존재하지 않습니다: {base_dir}")
            continue

        print(f"\n=== {base_dir} 내의 폴더 및 하위 폴더 크기 ===")

        # 첫 번째 레벨 폴더 크기 계산
        for dirpath, dirnames, filenames in os.walk(base_dir):
            # 현재 폴더 크기 계산
            folder_size = sum(
                os.path.getsize(os.path.join(dirpath, file))
                for file in filenames
            )
            print(f"{dirpath}: {folder_size / (1024 ** 2):.2f} MB")  # MB 단위로 출력

            # 하위 폴더 크기 계산
            for dirname in dirnames:
                subfolder_path = os.path.join(dirpath, dirname)
                subfolder_size = sum(
                    os.path.getsize(os.path.join(root, file))
                    for root, _, files in os.walk(subfolder_path)
                    for file in files
                )
                print(f"  {subfolder_path}: {subfolder_size / (1024 ** 2):.2f} MB")

            # 첫 레벨 폴더만 계산하고 하위 폴더로 들어가지 않음
            break
