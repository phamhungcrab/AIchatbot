# -------------------------------
# 🧠 Chatbot học tập cho môn Nhập môn Trí tuệ Nhân tạo (IT3160)
# File này là "main.py" – file chính khởi chạy ứng dụng Flask
# -------------------------------

# 📦 Import các thư viện cần thiết
from flask import Flask, render_template, request, redirect, url_for  # Flask framework để xây web app
import pandas as pd              # Xử lý dữ liệu dạng bảng
import pickle                    # Đọc file model đã lưu (Naive Bayes, KNN, vectorizer)
from preprocess import preprocess_text       # Hàm tiền xử lý văn bản (loại bỏ stopword, ký tự đặc biệt...)
from nb_module import predict_topic          # Hàm dự đoán chủ đề bằng mô hình Naïve Bayes
from find_answer import find_best_answer      # Hàm tìm câu trả lời gần nhất bằng KNN
from datastore import get_all_qa, get_qa_by_topic  # Các hàm truy xuất dữ liệu Q&A từ SQLite
import os                       # Thư viện thao tác với đường dẫn file/thư mục

# -------------------------------
# ⚙️ Thiết lập đường dẫn cho Flask
# -------------------------------

# BASE_DIR: đường dẫn tuyệt đối tới thư mục hiện tại (thư mục "app/")
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ROOT_DIR: lùi lên một cấp (thư mục cha chứa "app", "templates", "static"…)
ROOT_DIR = os.path.join(BASE_DIR, "..")

# -------------------------------
# 🚀 Khởi tạo ứng dụng Flask
# -------------------------------
app = Flask(
    __name__,
    template_folder=os.path.join(ROOT_DIR, "templates"),  # Thư mục chứa file .html (Jinja2 templates)
    static_folder=os.path.join(ROOT_DIR, "static")        # Thư mục chứa CSS, JS, ảnh, favicon, v.v.
)

# -------------------------------
# 📂 Nạp mô hình học máy đã huấn luyện sẵn
# -------------------------------

# vectorizer.pkl: mô hình chuyển văn bản thành vector số (TF-IDF, Bag-of-Words, v.v.)
with open('models/vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

# nb_model.pkl: mô hình Naïve Bayes → dùng để dự đoán chủ đề (topic)
with open('models/nb_model.pkl', 'rb') as f:
    nb_model = pickle.load(f)

# knn_model.pkl: mô hình KNN → dùng để tìm câu trả lời gần nhất trong tập câu hỏi cùng chủ đề
with open('models/knn_model.pkl', 'rb') as f:
    knn_model = pickle.load(f)

# -------------------------------
# 💬 Biến lưu lịch sử hội thoại
# -------------------------------
# Lưu tạm trong bộ nhớ RAM (dạng list), sẽ mất khi reload server
chat_history = []


# -------------------------------
# 🌐 ROUTE CHÍNH: Trang Chatbot
# -------------------------------
@app.route('/', methods=['GET', 'POST'])
def chatbot():
    global chat_history
    
    if request.method == 'POST':
        user_message = request.form['user_message']
        
        if user_message.strip():
            # Bước 1: Tiền xử lý
            processed = preprocess_text(user_message)
            
            # Bước 2: Dự đoán topic
            topic, topic_confidence = predict_topic(nb_model, vectorizer, processed)
            
            # Bước 3: Lấy câu hỏi trong topic
            df_topic = get_qa_by_topic(topic)
            
            # Bước 4: Tìm best match với threshold
            result = find_best_answer(
                vectorizer, 
                processed,  # ✅ Dùng processed thay vì user_message
                df_topic, 
                threshold=0.5  # ✅ Ngưỡng confidence tối thiểu
            )
            
            answer, question_similarity, matched_question = result
            
            # ✅ Tính final confidence
            if answer is None:
                # Không tìm thấy câu hỏi phù hợp
                final_confidence = 0.0
                answer = "Xin lỗi, tôi không tìm thấy câu trả lời phù hợp cho câu hỏi này."
            else:
                # Tính confidence tổng hợp
                final_confidence = (
                    0.10 * topic_confidence +      # 10% từ topic
                    0.60 * question_similarity +   # 60% từ question matching
                    0.30 * 0.8                     # 30% giả định các yếu tố khác = 0.8
                )
                
                # ✅ Thêm disclaimer dựa trên confidence
                if final_confidence >= 0.85:
                    pass  # Rất tin cậy, không cần disclaimer
                elif final_confidence >= 0.70:
                    answer += "\n\n💡 Nếu câu trả lời chưa chính xác, hãy hỏi chi tiết hơn."
                elif final_confidence >= 0.55:
                    answer += "\n\n⚠️ Tôi không hoàn toàn chắc chắn. Bạn có thể hỏi theo cách khác?"
                else:
                    answer = "🤔 Tôi không chắc lắm về câu trả lời này:\n\n" + answer
                    answer += "\n\n⚠️ Đề xuất: Hãy đặt câu hỏi rõ ràng hơn hoặc liên hệ giảng viên."
            
            # ✅ Lưu kèm confidence (optional - để debug/analysis)
            chat_history.append({
                "user": user_message,
                "bot": answer,
                "confidence": round(final_confidence, 3),
                "topic": topic,
                "topic_conf": round(topic_confidence, 3),
                "question_sim": round(question_similarity, 3) if question_similarity else 0.0
            })
        
        return redirect(url_for('chatbot'))
    
    return render_template('index.html', chat_history=chat_history)


# -------------------------------
# 🧹 ROUTE PHỤ: Xóa toàn bộ lịch sử chat
# -------------------------------
@app.route('/clear', methods=['POST'])
def clear_history():
    """
    Khi người dùng bấm nút 'Xóa lịch sử' → reset lại danh sách chat_history
    """
    global chat_history
    chat_history = []  # Làm trống danh sách hội thoại
    return redirect(url_for('chatbot'))  # Quay lại trang chatbot chính


# -------------------------------
# ▶️ Chạy Flask app
# -------------------------------
if __name__ == '__main__':
    # debug=True giúp auto reload khi thay đổi code và hiển thị log lỗi chi tiết
    app.run(debug=True)
