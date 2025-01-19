import cv2
import dlib
import numpy as np
import pandas as pd
import joblib

# 사전 훈련된 모델 경로 설정
#predictor_model = "C:\Users\PC\Desktop\graduate_project\project_facerecognition\dlib_face_recognition_resnet_model_v1.dat"
#face_descriptor_model = "C:\Users\PC\Desktop\graduate_project\project_facerecognition\shape_predictor_68_face_landmarks.dat"
predictor_model = "C:/Users/PC/Desktop/graduate_project/project_facerecognition/shape_predictor_68_face_landmarks.dat"
face_descriptor_model = "C:/Users/PC/Desktop/graduate_project/project_facerecognition/dlib_face_recognition_resnet_model_v1.dat"


# dlib의 얼굴 감지기 및 포즈 예측기 초기화
face_detector = dlib.get_frontal_face_detector()
face_pose_predictor = dlib.shape_predictor(predictor_model)
face_descriptor_extractor = dlib.face_recognition_model_v1(face_descriptor_model)

# 새로운 이미지 경로
image_path = 'test_img\is_img.png'

# 실제 레이블 딕셔너리
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

        # 훈련된 SVM 모델 로드
        model_path = 'classified_model\svm_face_recognition_model.pkl'
        svc = joblib.load(model_path)

        if embedding is not None:
            # SVM 모델에 테스트
            embedding = embedding.reshape(1, -1)  # 모델 입력 형식으로 변환
            predicted_label = svc.predict(embedding)

            probabilities = svc.predict_proba(embedding)
            predicted_prob = np.max(probabilities) * 100

            predicted_name = actual_labels[predicted_label[0]]

            # 결과 출력
            print(f'예측된 레이블: {predicted_name}')
            print(f'예측 확률: {predicted_prob:.2f}%')
        else:
            print("임베딩을 추출하지 못했습니다.")
