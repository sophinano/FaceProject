import numpy as np
import os
import cv2
import joblib
import tensorflow as tf
from sklearn.svm import SVC
from skimage import exposure # type: ignore
import dlib

# 사전 훈련된 모델 경로 설정
predictor_model = 'shape_predictor_68_face_landmarks.dat'
face_descriptor_model = 'dlib_face_recognition_resnet_model_v1.dat'
model_path = 'classified_model/keras_face_recognition_model.keras'
scaler_path = 'classified_model/scaler.pkl'
pca_svm_model_path = 'classified_model/pca_svm_face_recognition_model.pkl'
pca_rbf_svm_model_path = 'classified_model/pcarbf_svm_face_recognition_model.pkl'

# 실제 라벨 매핑
actual_labels = {
    0: '김우경',
    1: '김인서',
    2: '국소희',
    3: '안유진',
    4: '김채린'
}

# Keras 모델 및 스케일러 로드
pca_svm_model = joblib.load(pca_svm_model_path)
pca_rbf_svm_model = joblib.load(pca_rbf_svm_model_path)
loaded_model = tf.keras.models.load_model(model_path)
scaler = joblib.load(scaler_path)

# 얼굴 감지 및 임베딩 추출 함수
face_detector = dlib.get_frontal_face_detector()
face_pose_predictor = dlib.shape_predictor(predictor_model)
face_descriptor_extractor = dlib.face_recognition_model_v1(face_descriptor_model)

def preprocess_face(image, face_rect, target_size=(160, 160)):
    x1, y1, x2, y2 = face_rect.left(), face_rect.top(), face_rect.right(), face_rect.bottom()
    cropped_face = image[y1:y2, x1:x2]
    equalized_face = exposure.equalize_adapthist(cropped_face, clip_limit=0.03)
    resized_face = cv2.resize(equalized_face, target_size)
    normalized_face = resized_face / 255.0
    return normalized_face

def extract_face_embedding(image, face_rect):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    shape = face_pose_predictor(gray, face_rect)
    face_descriptor = face_descriptor_extractor.compute_face_descriptor(image, shape)
    return np.array(face_descriptor)

# 이미지에서 얼굴 인식 및 예측
def recognize_faces_in_images(image_folder, model, use_svm=False):
    image_files = [f for f in os.listdir(image_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    for image_file in image_files:
        image_path = os.path.join(image_folder, image_file)
        image = cv2.imread(image_path)

        if image is None:
            print(f"이미지 {image_file}을(를) 로드할 수 없습니다.")
            continue

        detected_faces = face_detector(image, 1)

        for idx, face_rect in enumerate(detected_faces):
            preprocessed_face = preprocess_face(image, face_rect)
            face_embedding = extract_face_embedding(image, face_rect)

            # 예측: SVM 또는 Keras 모델 선택
            if use_svm:
                prob = model.predict_proba([face_embedding])
                predicted_label = model.predict([face_embedding])
                predicted_name = actual_labels.get(predicted_label[0], "Unknown")
                predicted_prob = np.max(prob)  # 예측 확률
                print(f"Image: {image_file} | Face {idx + 1} | Predicted Name: {predicted_name} | Probability: {predicted_prob:.4f} | Model Used: SVM")
            else:
                face_embedding = scaler.transform([face_embedding])
                prob = loaded_model.predict(face_embedding)
                predicted_label = np.argmax(prob, axis=1)
                predicted_name = actual_labels.get(predicted_label[0], "Unknown")
                predicted_prob = np.max(prob)  # 예측 확률
                print(f"Image: {image_file} | Face {idx + 1} | Predicted Name: {predicted_name} | Probability: {predicted_prob:.4f} | Model Used: Keras")
# 카메라로 50장 사진 촬영 함수
def capture_images(num_images=50, output_dir='captured_images'):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    cap = cv2.VideoCapture(0)  # 카메라 열기
    captured_images = []

    for i in range(num_images):
        ret, frame = cap.read()
        if not ret:
            print("카메라 오류 발생")
            break

        # 이미지를 파일로 저장
        image_path = os.path.join(output_dir, f"img_{i+1}.jpg")
        cv2.imwrite(image_path, frame)
        captured_images.append(image_path)

        # 화면에 이미지를 표시 (선택적으로)
        cv2.imshow('Captured Image', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    return captured_images

captured_images = capture_images(num_images=50, output_dir='captured_images')
# 예시로 50장의 이미지에서 얼굴 예측하기
recognize_faces_in_images("captured_images", pca_svm_model, use_svm=True)  # SVM 모델 예시
recognize_faces_in_images("captured_images", pca_rbf_svm_model, use_svm=True)  # RBF SVM 모델 예시
recognize_faces_in_images("captured_images", loaded_model, use_svm=False)  # Keras 모델 예시