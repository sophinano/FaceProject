import cv2
import dlib
import numpy as np
import tensorflow as tf
import joblib
from PIL import ImageFont, ImageDraw, Image

# 사전 훈련된 모델 경로 설정
predictor_model = "C:/Users/PC/Desktop/graduate_project/project_facerecognition/shape_predictor_68_face_landmarks.dat"
face_descriptor_model = "C:/Users/PC/Desktop/graduate_project/project_facerecognition/dlib_face_recognition_resnet_model_v1.dat"

# dlib의 얼굴 감지기 및 포즈 예측기 초기화
face_detector = dlib.get_frontal_face_detector()
face_pose_predictor = dlib.shape_predictor(predictor_model)
face_descriptor_extractor = dlib.face_recognition_model_v1(face_descriptor_model)

# 실제 라벨 매핑
actual_labels = {
    0: '김우경',
    1: '김인서',
    2: '국소희',
    3: '안유진',
    4: '김채린'
}

# 훈련된 Keras 모델 및 스케일러 로드
model_path = 'classified_model/keras_face_recognition_model.keras'
scaler_path = 'classified_model/scaler.pkl'

loaded_model = tf.keras.models.load_model(model_path)
scaler = joblib.load(scaler_path)

# 웹캠 초기화 (DirectShow 백엔드 사용, 인덱스 0 시도)
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not cap.isOpened():
    print("웹캠을 열 수 없습니다.")
    exit()

print("웹캠을 시작합니다. 'q' 키를 눌러 종료하세요.")

# 폰트 설정 (한글을 지원하는 폰트 경로를 지정)
font_path = "C:/Windows/Fonts/malgun.ttf"  # Windows의 경우 '맑은 고딕' 경로
try:
    font = ImageFont.truetype(font_path, 20)
except IOError:
    print(f"폰트를 찾을 수 없습니다: {font_path}")
    exit()

# 임계값 설정 (예: 0.5)
CONFIDENCE_THRESHOLD = 0.8

while True:
    ret, frame = cap.read()
    if not ret:
        print("프레임을 가져올 수 없습니다.")
        break

    # 조명 보정 (CLAHE)
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced_frame = clahe.apply(gray_frame)

    # 얼굴 감지
    detected_faces = face_detector(enhanced_frame, 1)

    # PIL 이미지로 변환
    pil_image = Image.fromarray(enhanced_frame)
    draw = ImageDraw.Draw(pil_image)

    for face_rect in detected_faces:
        # 얼굴 영역의 좌표 추출
        x1, y1, x2, y2 = face_rect.left(), face_rect.top(), face_rect.right(), face_rect.bottom()

        # 사각형 그리기 (검은색 윤곽선)
        draw.rectangle([(x1, y1), (x2, y2)], outline=(0), fill=None)

        # 얼굴 랜드마크 예측 및 임베딩 추출
        shape = face_pose_predictor(enhanced_frame, face_rect)
        face_descriptor = face_descriptor_extractor.compute_face_descriptor(enhanced_frame, shape)
        embedding = np.array(face_descriptor)

        # 임베딩을 모델에 맞게 변환 및 예측
        embedding_reshaped = embedding.reshape(1, -1)
        embedding_scaled = scaler.transform(embedding_reshaped)
        prediction_prob = loaded_model.predict(embedding_scaled)
        predicted_class = np.argmax(prediction_prob, axis=1)[0]
        confidence = prediction_prob[0][predicted_class]

        # 임계값과 비교하여 라벨 결정
        if confidence >= CONFIDENCE_THRESHOLD:
            label = actual_labels.get(predicted_class, "Unknown")
        else:
            label = "Unknown"

        # 텍스트 준비
        text = f"{label}: {confidence:.2f}"

        # 텍스트 크기 계산
        text_bbox = draw.textbbox((x1 + 2, y1 - 28), text, font=font)
        text_size = (text_bbox[2] - text_bbox[0], text_bbox[3] - text_bbox[1])

        # 텍스트 배경 추가 (가독성 향상, 검은색)
        text_background = [(x1, y1 - 30), (x1 + text_size[0] + 4, y1 - 10)]
        draw.rectangle(text_background, fill=(0, 0, 0))  # 검은색 배경

        # 텍스트를 이미지에 그리기 (흰색)
        draw.text((x1 + 2, y1 - 28), text, font=font, fill=(255, 255, 255))  # 흰색 텍스트

    # PIL 이미지를 OpenCV 형식으로 변환
    frame_with_text = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

    # 프레임을 화면에 표시
    cv2.imshow('Real-Time Face Recognition', frame_with_text)

    # 'q' 키를 누르면 루프 종료
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 자원 해제
cap.release()
cv2.destroyAllWindows()
