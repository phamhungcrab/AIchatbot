import sqlite3
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, 'data', 'knowledge.db')

def add_question():
    print("📝 THÊM CÂU HỎI MỚI VÀO DATABASE")
    print("--------------------------------")
    
    question = input("Nhập câu hỏi: ").strip()
    if not question:
        print("❌ Câu hỏi không được để trống!")
        return

    answer = input("Nhập câu trả lời: ").strip()
    if not answer:
        print("❌ Câu trả lời không được để trống!")
        return
        
    topic = input("Nhập chủ đề (VD: AI, Python, General): ").strip()
    if not topic:
        topic = "General"

    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        # Kiểm tra xem câu hỏi đã tồn tại chưa
        cursor.execute("SELECT id FROM qa WHERE question = ?", (question,))
        if cursor.fetchone():
            print("⚠️ Câu hỏi này đã có trong database rồi!")
        else:
            cursor.execute("INSERT INTO qa (question, answer, topic) VALUES (?, ?, ?)", (question, answer, topic))
            conn.commit()
            print("✅ Đã thêm thành công!")
            print("💡 Lưu ý: Để Chatbot học được câu này, bạn cần chạy lại 'python app/augment_data.py' và 'python app/train_generative.py'.")
            
    except Exception as e:
        print(f"❌ Lỗi database: {e}")
    finally:
        conn.close()

if __name__ == "__main__":
    while True:
        add_question()
        cont = input("\nBạn có muốn thêm câu khác không? (y/n): ").lower()
        if cont != 'y':
            break
