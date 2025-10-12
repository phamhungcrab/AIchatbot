# -------------------------------
# 📘 datastore.py — Tầng dữ liệu (Data Layer) của Chatbot
# Chức năng: kết nối, truy vấn, và khởi tạo cơ sở dữ liệu SQLite
# -------------------------------

import sqlite3     # Thư viện chuẩn của Python để làm việc với SQLite
import os          # Dùng để xử lý đường dẫn file, thư mục
import pandas as pd  # Thư viện mạnh để xử lý dữ liệu dạng bảng (DataFrame)

# -------------------------------
# 📍 Xác định đường dẫn đến cơ sở dữ liệu knowledge.db
# -------------------------------
# os.path.dirname(__file__) → thư mục hiện tại chứa file datastore.py (thường là "app/")
# ".." → lùi lên thư mục cha (AIChatbot/)
# "data/knowledge.db" → đường dẫn đến file database thật sự
DB_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "knowledge.db")


# =========================================================
# 📂 1️⃣ HÀM LẤY TOÀN BỘ DỮ LIỆU Q&A
# =========================================================
def get_all_qa():
    """
    Lấy toàn bộ dữ liệu gồm (question, answer, topic)
    từ cơ sở dữ liệu knowledge.db.
    Trả về dạng bảng (pandas DataFrame) để dễ xử lý.
    """
    # Kết nối đến SQLite bằng context manager (with)
    # Khi khối with kết thúc, kết nối sẽ tự đóng lại — an toàn, gọn gàng.
    with sqlite3.connect(DB_PATH) as conn:
        # Dùng pandas đọc truy vấn SQL trực tiếp thành DataFrame
        df = pd.read_sql_query('SELECT question, answer, topic FROM qa', conn)
    return df


# =========================================================
# 📂 2️⃣ HÀM LẤY DỮ LIỆU THEO CHỦ ĐỀ
# =========================================================
def get_qa_by_topic(topic):
    """
    Lấy danh sách câu hỏi và câu trả lời thuộc một chủ đề cụ thể (topic).
    Trả về DataFrame có 2 cột: question, answer.
    """
    with sqlite3.connect(DB_PATH) as conn:
        # Dấu ? trong câu SQL là placeholder — giúp tránh lỗi SQL injection.
        query = 'SELECT question, answer FROM qa WHERE topic = ?'
        df = pd.read_sql_query(query, conn, params=(topic,))
    return df


# =========================================================
# ✏️ 3️⃣ HÀM THÊM DỮ LIỆU MỚI VÀO BẢNG
# =========================================================
def insert_qa(question, answer, topic):
    """
    Thêm một cặp (question, answer, topic) mới vào cơ sở dữ liệu.
    """
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()  # Tạo con trỏ để thực thi lệnh SQL
        cursor.execute(
            'INSERT INTO qa (question, answer, topic) VALUES (?, ?, ?)',
            (question, answer, topic)
        )
        conn.commit()  # Lưu thay đổi vào DB (nếu không commit sẽ không lưu thật)


# =========================================================
# 🏗️ 4️⃣ HÀM KHỞI TẠO BẢNG Q&A
# =========================================================
def init_db():
    """
    Tạo bảng 'qa' trong cơ sở dữ liệu nếu chưa tồn tại.
    (Dùng khi chạy lần đầu tiên để tạo database.)
    """
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        # Câu lệnh SQL tạo bảng với 4 cột:
        # id (tự tăng), question, answer, topic
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS qa (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                question TEXT NOT NULL,
                answer TEXT NOT NULL,
                topic TEXT NOT NULL
            );
        ''')
        conn.commit()

        # 🧹 Nếu muốn xóa bảng để tạo lại hoàn toàn từ đầu, có thể bỏ comment 2 dòng sau:
        # cursor.execute("DROP TABLE qa")
        # cursor.execute("DROP TABLE sqlite_sequence")
        # conn.commit()


# =========================================================
# 🚀 5️⃣ KHỐI MAIN — chỉ chạy khi gọi file này trực tiếp
# =========================================================
if __name__ == '__main__':
    """
    Khi chạy file này trực tiếp bằng lệnh:
        python app/datastore.py
    → chương trình sẽ tạo database knowledge.db và nạp dữ liệu mẫu từ file init.sql
    """
    # Xác định lại đường dẫn để tạo DB và đọc file SQL
    BASE_DIR = os.path.dirname(__file__)
    DATA_DIR = os.path.join(BASE_DIR, "..", "data")
    # os.makedirs(DATA_DIR, exist_ok=True)  # Tạo thư mục data nếu chưa có

    DB_PATH = os.path.join(DATA_DIR, "knowledge.db")  # đường dẫn database
    SQL_PATH = os.path.join(DATA_DIR, "init.sql")     # file SQL chứa dữ liệu mẫu

    # --- BƯỚC 2️⃣: đọc toàn bộ nội dung file init.sql ---
    with open(SQL_PATH, "r", encoding="utf-8") as f:
        sql_script = f.read()

    # --- BƯỚC 3️⃣: kết nối và thực thi toàn bộ script SQL ---
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.executescript(sql_script)  # Thực thi nhiều lệnh SQL liên tiếp trong file
        conn.commit()

    # In ra console để báo thành công
    print("✅ Database 'knowledge.db' created and sample data inserted successfully!")
