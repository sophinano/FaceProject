# attend.py
from flask import Flask, render_template, Response, redirect, url_for, jsonify
import cv2
import dlib
import numpy as np
import tensorflow as tf
import joblib
import sqlite3
from sklearn.preprocessing import StandardScaler
from PIL import ImageFont, ImageDraw, Image
import time
from dlib import correlation_tracker

app = Flask(__name__)

db_path = 'D:\\SEPJ\\project_facerecognition\\university.db' 

# 사전 훈련된 모델 경로 설정
predictor_model = 'D:\\SEPJ\\project_facerecognition\\classified_model\\shape_predictor_68_face_landmarks.dat'
face_descriptor_model = 'D:\\SEPJ\\project_facerecognition\\classified_model\\dlib_face_recognition_resnet_model_v1.dat'
model_path = 'D:\\SEPJ\\project_facerecognition\\classified_model\\keras_face_recognition_model.keras'
scaler_path = 'D:\\SEPJ\\project_facerecognition\\classified_model\\scaler.pkl'

# dlib의 얼굴 감지기 및 포즈 예측기 초기화
face_detector = dlib.get_frontal_face_detector()
face_pose_predictor = dlib.shape_predictor(r'D:\SEPJ\project_facerecognition\classified_model\shape_predictor_68_face_landmarks.dat')
face_descriptor_extractor = dlib.face_recognition_model_v1(r'D:\SEPJ\project_facerecognition\classified_model\dlib_face_recognition_resnet_model_v1.dat')

# 실제 라벨 매핑
actual_labels = {
   0: '김우경',
   1: '김인서',
   2: '국소희',
   3: '안유진',
   4: '김채린'
}

# Keras 모델 및 스케일러 로드
loaded_model = tf.keras.models.load_model(model_path)
scaler = joblib.load(scaler_path)
# 폰트 설정
font_path = "C:/Windows/Fonts/malgun.ttf"
try:
    font = ImageFont.truetype(font_path, 20)
except IOError:
    print(f"폰트를 찾을 수 없습니다: {font_path}")
    exit()

# 비디오 캡처 초기화
video_capture = cv2.VideoCapture(0)  # 기본 카메라 사용

#전역변수 추가
is_running=True 
start_time = None # 캠 시작 시간을 저장할 변수
tracker = correlation_tracker()
tracked_students = {}  # 학생 이름과 트래킹 상태를 저장하는 딕셔너리
TRACK_TIMEOUT = 10  # 얼굴 인식이 안 된 후 경과 시간 (10초로 설정)


def record_attendance(student_id, class_id, status="출석"):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO attendance (student_id, class_id, status)
        VALUES (?, ?, ?)
    ''', (student_id, class_id, status))
    conn.commit()
    conn.close()

def get_student_id_by_name(name):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute('''
        SELECT studentID FROM users WHERE username = ?
    ''', (name,))
    student_id = cursor.fetchone()
    conn.close()
    return student_id[0] if student_id else None


def get_current_class_id():
    return 1 #예시 = 1


def generate_frames():
    global is_running, start_time, tracker, tracked_students

    if not video_capture.isOpened():
        print("카메라를 열 수 없습니다")
        return

    start_time = time.time()  # 캠 시작 시간을 기록합니다.

    while is_running:
        success, frame = video_capture.read()
        if not success:
            break
        else:
            # 시간 체크: 20초가 경과했는지 확인
            if time.time() - start_time >= 20:
                is_running = False  # 30초가 지나면 is_running을 False로 설정

            rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            detected_faces = face_detector(rgb_image, 1)
            current_time = time.time()

            if len(detected_faces) > 0:
                # 얼굴이 감지된 경우: 첫 번째 얼굴만 처리하는 예시
                face_rect = detected_faces[0]  # 첫 번째 얼굴
                x1, y1, x2, y2 = face_rect.left(), face_rect.top(), face_rect.right(), face_rect.bottom()

                # 트래커 초기화 및 갱신
                tracker.start_track(rgb_image, face_rect)

                # 얼굴 특징점 예측
                shape = face_pose_predictor(rgb_image, face_rect)
                face_descriptor = face_descriptor_extractor.compute_face_descriptor(rgb_image, shape)
                embedding = np.array(face_descriptor).reshape(1, -1)

                # 데이터 정규화 및 예측
                new_embedding_scaled = scaler.transform(embedding)
                prediction_prob = loaded_model.predict(new_embedding_scaled)
                predicted_class = np.argmax(prediction_prob, axis=1)[0]
                confidence = prediction_prob[0][predicted_class]
                label = actual_labels.get(predicted_class, "Unknown")

                # 트래킹 상태 기록: 얼굴이 인식된 학생 저장
                if label != "Unknown":
                    tracked_students[label] = current_time  # 학생 이름과 마지막 인식 시간 기록
                    student_id = get_student_id_by_name(label)
                    class_id = get_current_class_id()
                    record_attendance(student_id, class_id, "출석")

                # 얼굴 영역 및 정보 표시
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                text = f"{label}: {confidence:.2f}"
                pil_image = Image.fromarray(frame)
                draw = ImageDraw.Draw(pil_image)
                draw.text((x1 + 2, y1 - 28), text, font=font, fill=(0, 0, 0, 255))
                frame = np.array(pil_image)

            # 10초 동안 인식되지 않은 학생을 확인
            for student, last_seen_time in list(tracked_students.items()):
                if current_time - last_seen_time > TRACK_TIMEOUT:
                    print(f"{student} 학생이 인식되지 않습니다.")
                    # 학생을 추적에서 제외 (삭제)
                    del tracked_students[student]

            # 프레임을 JPEG 형식으로 인코딩
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()

            # 프레임을 멀티파트 형식으로 전송
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    video_capture.release()  # 비디오 캡처 해제



@app.route('/')
def index():
    return redirect(url_for('attendance'))

@app.route('/attendance')
def attendance():
    return render_template('attendance.html')

@app.route('/attendancecheck')
def attendancecheck():
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute('''
        SELECT a.attendance_id, u.username, c.class_name, a.attendance_date, a.status
        FROM attendance a
        JOIN users u ON a.student_id = u.studentID
        JOIN classes c ON a.class_id = c.class_id
        ORDER BY a.attendance_date DESC
    ''')
    record_attendance = cursor.fetchall()
    conn.close()
    return render_template('attendancecheck.html', records=record_attendance)  # 출결 페이지 렌더링

@app.route('/get_attendance')
def get_attendance():
    # 출석한 학생의 이름을 리스트 형태로 반환
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute('SELECT username FROM attendance JOIN users ON attendance.student_id = users.studentID WHERE status = "출석"')
    students = cursor.fetchall()
    conn.close()

    # 학생 이름만 추출하여 리스트로 반환
    student_names = list(set(student[0] for student in students))
    return jsonify(student_names)

# 종료 라우트에서 is_running을 False로 설정
@app.route('/stop', methods=['POST'])
def stop():
    global is_running
    is_running = False
    return jsonify(success=True)

@app.route('/account')
def account():
    return render_template('account.html') 

@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html')  # 대시보드 페이지 렌더링

@app.route('/calendar')
def calendar():
    return render_template('calendar.html')  # 캘린더 페이지 렌더링

@app.route('/studentlist')
def studentlist():
    return render_template('studentlist.html')  # 학생 조회 페이지 렌더링

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True, port=5001)
