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
from knn_module import find_best_answer      # Hàm tìm câu trả lời gần nhất bằng KNN
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
    """
    Xử lý 2 trường hợp:
    - GET: hiển thị giao diện chatbot cùng lịch sử trò chuyện
    - POST: nhận câu hỏi từ người dùng, xử lý và tạo phản hồi
    """
    global chat_history  # Sử dụng biến toàn cục để lưu lịch sử hội thoại

    # Khi người dùng gửi tin nhắn từ form HTML
    if request.method == 'POST':
        user_message = request.form['user_message']  # Lấy nội dung người dùng nhập

        # Kiểm tra tin nhắn không rỗng
        if user_message.strip():
            # 🧹 Bước 1: Tiền xử lý văn bản (chuẩn hóa, xóa ký tự đặc biệt, chuyển thường,...)
            processed = preprocess_text(user_message)

            # 🧩 Bước 2: Dự đoán chủ đề (topic) bằng mô hình Naïve Bayes
            # predict_topic trả về (tên_chủ_đề, độ_tin_cậy)
            topic, confidence = predict_topic(nb_model, vectorizer, processed)

            # 🗂️ Bước 3: Lấy các câu hỏi - câu trả lời cùng chủ đề từ database
            df_topic = get_qa_by_topic(topic)

            # 🔍 Bước 4: Tìm câu trả lời gần nhất với câu hỏi người dùng bằng KNN
            answer = find_best_answer(knn_model, vectorizer, user_message, df_topic)

            # Nếu không tìm thấy câu trả lời phù hợp thì phản hồi mặc định
            if not answer:
                answer = "Xin lỗi, tôi chưa có thông tin về câu hỏi này."

            # 📝 Lưu hội thoại (user hỏi - bot trả lời) vào danh sách lịch sử
            chat_history.append({"user": user_message, "bot": answer})

        # Sau khi xử lý xong → quay lại route "/" để hiển thị tin nhắn mới
        return redirect(url_for('chatbot'))

    # Nếu là GET → hiển thị trang index.html cùng lịch sử hội thoại
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
