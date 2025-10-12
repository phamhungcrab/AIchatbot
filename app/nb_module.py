# -------------------------------
# 🤖 nb_module.py — Mô-đun Naïve Bayes cho Chatbot
# Chức năng: dự đoán chủ đề (topic) của câu hỏi đầu vào
# và huấn luyện mô hình Naïve Bayes lưu vào file nb_model.pkl
# -------------------------------

import numpy as np
import pickle
import os

# -------------------------------
# 📁 Thiết lập đường dẫn lưu mô hình
# -------------------------------
BASE_DIR = os.path.dirname(__file__)                   # → thư mục hiện tại ("app/")
MODEL_DIR = os.path.join(BASE_DIR, '..', 'models')     # → lùi lên một cấp (AIChatbot/models)
# os.makedirs(MODEL_DIR, exist_ok=True)                # Có thể bật lại nếu muốn tự tạo thư mục
MODEL_PATH = os.path.join(MODEL_DIR, 'nb_model.pkl')   # Đường dẫn file model Naïve Bayes

# =========================================================
# 🔮 1️⃣ HÀM DỰ ĐOÁN CHỦ ĐỀ CÂU HỎI (SỬ DỤNG MÔ HÌNH ĐÃ HUẤN LUYỆN)
# =========================================================
def predict_topic(nb_model, vectorizer, text):
    """
    ✅ Mục đích:
        Dự đoán chủ đề (topic) của câu hỏi người dùng bằng mô hình Naïve Bayes.

    📌 Tham số:
        nb_model   : mô hình Naïve Bayes đã được load sẵn từ file nb_model.pkl
        vectorizer : mô hình TF-IDF hoặc CountVectorizer (đã huấn luyện cùng model)
        text       : câu hỏi người dùng (chuỗi string)

    🔁 Trả về:
        (predicted_topic, confidence)
        - predicted_topic: tên chủ đề được dự đoán (ví dụ: "MachineLearning")
        - confidence: độ tin cậy của dự đoán (xác suất lớn nhất)
    """

    # 🧩 1. Biến đổi văn bản đầu vào thành vector TF-IDF
    # Vectorizer đã được huấn luyện sẵn trên dữ liệu Q&A, giúp mô hình hiểu "ngữ nghĩa" cơ bản
    X = vectorizer.transform([text])

    # 🧠 2. Dự đoán nhãn chủ đề bằng mô hình Naïve Bayes
    predicted_topic = nb_model.predict(X)[0]

    # 📈 3. Tính xác suất dự đoán cho tất cả các chủ đề
    probs = nb_model.predict_proba(X)
    confidence = np.max(probs)  # Lấy giá trị xác suất cao nhất làm độ tin cậy

    # 💬 4. Trả về kết quả (chủ đề, độ tin cậy)
    return predicted_topic, round(float(confidence), 4)


# =========================================================
# 🧠 2️⃣ HÀM HUẤN LUYỆN MÔ HÌNH NAÏVE BAYES (chạy offline)
# =========================================================
def train_naive_bayes(vectorizer, train_texts, train_labels):
    """
    ✅ Mục đích:
        Huấn luyện mô hình Naïve Bayes để phân loại câu hỏi vào các chủ đề (topics)
        và lưu lại model đã huấn luyện vào file nb_model.pkl.

    📌 Tham số:
        vectorizer   : mô hình TF-IDF hoặc CountVectorizer (đã fit sẵn)
        train_texts  : danh sách các câu hỏi huấn luyện (list[str])
        train_labels : danh sách nhãn chủ đề tương ứng (list[str])

    🔁 Trả về:
        nb_model: mô hình Naïve Bayes đã huấn luyện
    """

    # ==============================
    # ⚙️ Cách triển khai đầy đủ (đã comment sẵn):
    # - Có thể tinh chỉnh vectorizer, chọn tham số alpha, kiểm thử bằng cross-validation
    # - Để giữ code đơn giản, phần đó được ẩn đi (bạn có thể mở lại khi cần)
    # ==============================

    # from sklearn.naive_bayes import MultinomialNB
    # from sklearn.model_selection import cross_val_score
    # import numpy as np
    #
    # # (1) Tùy chỉnh vectorizer (nếu cần)
    # vectorizer.set_params(
    #     max_features=800,        # Giới hạn số đặc trưng để tránh ma trận quá thưa
    #     ngram_range=(1, 2),      # Học cả từ đơn (unigram) và cụm 2 từ (bigram)
    #     min_df=1                 # Giữ từ xuất hiện ít nhất 1 lần
    # )
    #
    # X_train = vectorizer.fit_transform(train_texts)
    #
    # # (2) Huấn luyện mô hình Multinomial Naïve Bayes
    # nb_model = MultinomialNB(alpha=1.2, fit_prior=True)
    # nb_model.fit(X_train, train_labels)
    #
    # # (3) Đánh giá sơ bộ bằng cross-validation
    # scores = cross_val_score(nb_model, X_train, train_labels, cv=5)
    # print(f"📊 Cross-val accuracy: {np.mean(scores):.3f} ± {np.std(scores):.3f}")

    # ==============================
    # 🧩 Phiên bản đơn giản được dùng thực tế trong chatbot:
    # ==============================
    from sklearn.naive_bayes import MultinomialNB

    # Biến đổi dữ liệu huấn luyện thành vector TF-IDF
    X_train = vectorizer.transform(train_texts)

    # Khởi tạo mô hình Naïve Bayes dạng Multinomial (phù hợp với dữ liệu văn bản)
    nb_model = MultinomialNB()

    # Huấn luyện mô hình với dữ liệu huấn luyện
    nb_model.fit(X_train, train_labels)

    # 💾 Lưu mô hình đã huấn luyện lại để dùng sau
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(nb_model, f)

    # Thông báo khi lưu thành công
    print("✅ Model with Naïve Bayes saved at:", os.path.abspath(MODEL_PATH))

    return nb_model
