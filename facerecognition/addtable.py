import sqlite3

# 통합된 데이터베이스 파일 생성
db_path = 'D:\\SEPJ\\project_facerecognition\\university.db'

def add_reply_to_field():
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # messages 테이블에 reply_to_message_id 필드가 없는 경우 추가
    cursor.execute('''
        ALTER TABLE messages
        ADD COLUMN reply_to_message_id INTEGER REFERENCES messages(message_id)
    ''')
    
    conn.commit()
    conn.close()

def create_attendance():
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

     # 기존 attendance 테이블이 있으면 삭제
    cursor.execute('DROP TABLE IF EXISTS attendance')

    # 주차와 교시 정보를 포함한 새로운 테이블 생성
    cursor.execute('''
        CREATE TABLE attendance (
            attendance_id INTEGER PRIMARY KEY AUTOINCREMENT,
            student_id VARCHAR(50),
            class_id INTEGER,
            week INTEGER,     -- 주차 정보
            period INTEGER,   -- 교시 정보
            attendance_date DATE DEFAULT (DATE('now')),
            status VARCHAR(10) DEFAULT '미정',
            FOREIGN KEY (student_id) REFERENCES student(studentID),
            FOREIGN KEY (class_id) REFERENCES classes(class_id),
            UNIQUE(student_id, class_id, week, period)
)
    ''')
    conn.commit()
    conn.close()

#add_reply_to_field()
create_attendance()

