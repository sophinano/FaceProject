import sqlite3


db_path='D:\\SEPJ\\project_facerecognition\\university.db'
def fun():
    datas = [
        ('21018017','1','출석'),
        ('21018004','1','출석'),
        ('21018049','1','출석'),
        ('21018018','1','출석'),
        ('21018022','1','출석'),
        ('21018018','1','2024-10-26','출석'),
        ('21018018','1','2024-11-04','출석'),
    ]
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.executemany('''
                    INSERT INTO attendance (student_id,class_id,status)
                    VALUES (?,?,?)
                    ''', datas)

    conn.commit()
    conn.close()

def insert_attendance_data():
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # 데이터 삽입
    attendance_data = [
        # '21018018'
        ('21018018', 1, '2024-10-17', '출석'),
        ('21018018', 1, '2024-10-24', '출석'),
        ('21018018', 1, '2024-10-31', '출석'),
        ('21018018', 1, '2024-11-07', '출석'),
        
        # '21018017'
        ('21018017', 1, '2024-10-17', '출석'),
        ('21018017', 1, '2024-10-24', '출석'),
        ('21018017', 1, '2024-10-31', '출석'),
        ('21018017', 1, '2024-11-07', '출석'),

        # '21018004'
        ('21018004', 1, '2024-10-17', '출석'),
        ('21018004', 1, '2024-10-24', '출석'),
        ('21018004', 1, '2024-10-31', '출석'),
        ('21018004', 1, '2024-11-07', '출석'),

        # '21018049'
        ('21018049', 1, '2024-10-17', '출석'),
        ('21018049', 1, '2024-10-24', '출석'),
        ('21018049', 1, '2024-10-31', '출석'),
        ('21018049', 1, '2024-11-07', '출석'),

        # '21018022'
        ('21018022', 1, '2024-10-17', '출석'),
        ('21018022', 1, '2024-10-24', '출석'),
        ('21018022', 1, '2024-10-31', '출석'),
        ('21018022', 1, '2024-11-07', '출석'),
    ]

    
    cursor.executemany('''
        INSERT INTO attendance (student_id, class_id, attendance_date, status)
        VALUES (?, ?, ?, ?)
    ''', attendance_data)
    
    conn.commit()
    conn.close()


def fun2():
    enrollment_data = [
        ('21018017','1'),
        ('21018017','7'),
        ('21018017','8'),
        ('21018004','1'),
        ('21018004','2'),
        ('21018004','7'),
        ('21018049','1'),
        ('21018049','5'),
        ('21018049','3'),
        ('21018018','1'),
        ('21018018','4'),
        ('21018018','5'),
        ('21018022','1'),
        ('21018022','2'),
        ('21018022','8'),
    ]
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.executemany('''
                    INSERT INTO enrollment (student_id,class_id)
                    VALUES (?,?)
                    ''', enrollment_data)
    conn.commit()
    conn.close()


def fun1():
    datas = [
        ('21018017','1','2024-10-17','출석'),
        ('21018004','1','2024-10-17','출석'),
        ('21018049','1','2024-10-17','출석'),
        ('21018018','1','2024-10-17','출석'),
        ('21018022','1','2024-10-17','출석'),
    ]
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.executemany('''
                    INSERT INTO attendance (student_id,class_id,attendance_date,status)
                    VALUES (?,?,?,?)
                    ''', datas)

    conn.commit()
    conn.close()

insert_attendance_data()