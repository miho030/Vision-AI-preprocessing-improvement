import os

""" npy, h5 파일 가져올 폴더 생성 """
folders = ["AutoEncoder", "PCA"]
current_path = os.getcwd()
print(f"현재 경로: {current_path}")

# 폴더 생성
for folder in folders:
    folder_path = os.path.join(current_path, folder)
    try:
        os.makedirs(folder_path, exist_ok=True)
        print(f"폴더 생성 완료: {folder_path}")
    except Exception as e:
        print(f"폴더 생성 실패: {folder_path}, 오류: {e}")

