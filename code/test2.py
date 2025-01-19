import cv2
import dlib
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import joblib
# 사전 훈련된 모델 경로 설정
predictor_model = "C:/Users/PC/Desktop/graduate_project/project_facerecognition/shape_predictor_68_face_landmarks.dat"
face_descriptor_model = "C:/Users/PC/Desktop/graduate_project/project_facerecognition/dlib_face_recognition_resnet_model_v1.dat"

# dlib의 얼굴 감지기 및 포즈 예측기 초기화
face_detector = dlib.get_frontal_face_detector()
face_pose_predictor = dlib.shape_predictor(predictor_model)
face_descriptor_extractor = dlib.face_recognition_model_v1(face_descriptor_model)

# 새로운 이미지 경로
image_path = 'test_img\suho.jpg'

actual_labels = {
   0: '김우경',
   1: '김인서',
   2: '국소희',
   3: '안유진',
   4: '김채린'
}

# 이미지 로드
image = cv2.imread(image_path)
if image is None:
    print(f"Could not load image: {image_path}")
else:
    # 이미지 RGB로 변환
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # 얼굴 감지
    detected_faces = face_detector(rgb_image, 1)

    if len(detected_faces) == 0:
        print("No faces found in the image.")
    else:
        # 첫 번째 얼굴에 대한 임베딩 추출
        face_rect = detected_faces[0]
        shape = face_pose_predictor(rgb_image, face_rect)
        face_descriptor = face_descriptor_extractor.compute_face_descriptor(rgb_image, shape)

        # 임베딩을 NumPy 배열로 변환
        embedding = np.array(face_descriptor)

# 훈련된 Keras 모델 로드
model_path = 'classified_model\keras_face_recognition_model.keras'
scaler_path='classified_model\scaler.pkl'
#데이터 정규화
scaler = joblib.load(scaler_path)
# 예측 예제
loaded_model = tf.keras.models.load_model(model_path)
embedding = embedding.reshape(1, -1)
new_embedding_scaled = scaler.transform(embedding)  # 정규화 적용
prediction_prob = loaded_model.predict(new_embedding_scaled)
predicted_class = np.argmax(prediction_prob, axis=1)
print(f'예측된 클래스: {predicted_class[0]}')
print('예측 확률:')
for class_name, prob in zip(actual_labels, prediction_prob[0]):
    print(f'  {class_name}: {prob:.2f}')