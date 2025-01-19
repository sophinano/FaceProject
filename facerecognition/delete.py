import sqlite3
db_path = 'D:\\SEPJ\\project_facerecognition\\university.db'
def clear_attendance_records():
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute('DELETE FROM attendance')
    #cursor.execute('DELETE FROM department')
    #cursor.execute('DELETE FROM messages')
    conn.commit()
    conn.close()

clear_attendance_records()