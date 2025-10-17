# -------------------------------
# 🧠 train_models.py — Huấn luyện toàn bộ mô hình cho Chatbot
# Chức năng:
#   - Đọc dữ liệu Q&A từ cơ sở dữ liệu SQLite (knowledge.db)
#   - Tiền xử lý văn bản (chuẩn hóa ngôn ngữ)
#   - Huấn luyện TF-IDF vectorizer, mô hình Naïve Bayes, và KNN
#   - Lưu các mô hình ra thư mục "models/"
# -------------------------------

import pandas as pd
import pickle
from datastore import get_all_qa                  # Lấy dữ liệu Q&A từ database
from preprocess import preprocess_text, train_vectorizer
from nb_module import train_naive_bayes
from knn_module import train_knn

import nltk
import os

# -------------------------------
# 📦 TẢI DỮ LIỆU HỖ TRỢ TỪ NLTK (lần đầu tiên chạy)
# -------------------------------
# Các gói này giúp tokenization, stopwords filtering và lemmatization hoạt động chính xác.
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt_tab')   # (bổ sung để hỗ trợ một số trường hợp đặc biệt trong NLTK)

# -------------------------------
# 📁 Thiết lập đường dẫn thư mục lưu mô hình
# -------------------------------
BASE_DIR = os.path.dirname(__file__)               # → thư mục hiện tại ("app/")
MODEL_DIR = os.path.join(BASE_DIR, '..', 'models') # → lùi 1 cấp đến thư mục "models/"
# os.makedirs(MODEL_DIR, exist_ok=True)            # Bật lại nếu cần tự động tạo thư mục

# =========================================================
# 🚀 HÀM CHÍNH: HUẤN LUYỆN TOÀN BỘ MÔ HÌNH CHATBOT
# =========================================================
def train_all_models():
    """
    ✅ Mục đích:
        - Đọc toàn bộ dữ liệu Q&A từ knowledge.db
        - Tiền xử lý văn bản
        - Huấn luyện TF-IDF, Naïve Bayes, và KNN
        - Lưu mô hình ra thư mục models/

    🔁 Kết quả:
        models/
        ├── vectorizer.pkl
        ├── nb_model.pkl
        └── knn_model.pkl
    """

    # ------------------------------------
    # 1️⃣ Đọc dữ liệu từ database
    # ------------------------------------
    print('📚 Đang đọc dữ liệu từ knowledge.db...')
    df = get_all_qa()  # Trả về DataFrame gồm [question, answer, topic]

    # Kiểm tra dữ liệu có trống không
    if df.empty:
        print('⚠️ Không có dữ liệu trong cơ sở dữ liệu! Kiểm tra knowledge.db.')
        return

    # ------------------------------------
    # 2️⃣ Tiền xử lý dữ liệu văn bản
    # ------------------------------------
    # Gọi hàm preprocess_text() cho từng câu hỏi
    #   - Chuyển chữ thường
    #   - Xóa ký tự đặc biệt, số, stopwords
    #   - Tokenize lại thành văn bản sạch
    print('🧹 Đang tiền xử lý dữ liệu...')
    df['clean_text'] = df['question'].apply(preprocess_text)

    # ------------------------------------
    # 3️⃣ Huấn luyện TF-IDF vectorizer
    # ------------------------------------
    # TF-IDF sẽ chuyển từng câu hỏi thành vector số học (đặc trưng)
    print('⚙️ Đang huấn luyện TF-IDF vectorizer...')
    vectorizer = train_vectorizer(df['clean_text'])

    # Lưu TF-IDF đã huấn luyện để dùng lại khi dự đoán
    with open(os.path.join(MODEL_DIR, 'vectorizer.pkl'), 'wb') as f:
        pickle.dump(vectorizer, f)

    # ------------------------------------
    # 4️⃣ Huấn luyện mô hình Naïve Bayes
    # ------------------------------------
    # Mô hình này học cách phân loại câu hỏi vào các chủ đề (topics)
    print('🧠 Đang huấn luyện mô hình Naïve Bayes...')
    nb_model = train_naive_bayes(vectorizer, df['clean_text'], df['topic'])

    # ------------------------------------
    # 5️⃣ Huấn luyện mô hình KNN
    # ------------------------------------
    # Mô hình này dùng để tìm câu hỏi tương tự nhất → chọn câu trả lời gần nhất
    print('🔍 Đang huấn luyện mô hình KNN...')
    knn_model = train_knn(
        vectorizer,                # Bộ vector hóa TF-IDF
        df['clean_text'],          # Dữ liệu huấn luyện
        df['topic'],               # Nhãn chủ đề (topic)
        n_neighbors=8              # Số lượng láng giềng gần nhất (k)
    )

    # ------------------------------------
    # 6️⃣ Kết thúc quá trình huấn luyện
    # ------------------------------------
    print('✅ Hoàn tất huấn luyện!')
    print('📦 Các mô hình đã được lưu trong thư mục: models/')
    print('   ├── vectorizer.pkl')
    print('   ├── nb_model.pkl')
    print('   └── knn_model.pkl')


# =========================================================
# ▶️ CHẠY TRỰC TIẾP FILE
# =========================================================
if __name__ == '__main__':
    # Khi chạy file bằng lệnh:
    #   python app/train_models.py
    # → Toàn bộ quy trình huấn luyện sẽ được thực hiện
    train_all_models()
