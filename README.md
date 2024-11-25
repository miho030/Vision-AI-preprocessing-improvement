## PCA/AutoEncoder + CNN
#AutoEncoder+CNN(InceptionV3)_32x32
#AutoEncoder+CNN(InceptionV3)_64x64
#AutoEncoder+CNN(MobileNet)_32x32
#AutoEncoder+CNN(MobileNet)_64x64

#PCA+CNN(InceptionV3)_32x32
#PCA+CNN(InceptionV3)_64x64
#PCA+CNN(MobileNet)_32x32
#PCA+CNN(MobileNet)_64x64

###fogging
## 학습 포함 모델
#PCA(MobileNet)_32x32
#AutoEncoder(MobileNet)_32x32
#CNN(MobileNet)_32x32

## 학습 미 포함 (우선 save_ 파일에서 모델 파일을 생성, load_ 파일에서 빠르게 실행)
#실행순서
#save_AutoEncoder(MobileNet)_32x32		# 축소 모델을 생성하기 위한 파일
#Load_AutoEncoder(MobileNet)_32x32	# 저장된 축소 모델을 이용해서 축소 데이터 생성
#save_CNN(MobileNet)		# 분류 모델 구조 저장
#load_CNN(MobileNet)	# 저장된 축소 데이터를 저장된 분류 모델로 평가