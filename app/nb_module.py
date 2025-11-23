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
class CustomMultinomialNB:
    """
    Tự cài đặt thuật toán Multinomial Naive Bayes (tương tự sklearn).
    """
    def __init__(self, alpha=1.0):
        self.alpha = alpha
        self.class_log_prior_ = None
        self.feature_log_prob_ = None
        self.classes_ = None

    def fit(self, X, y):
        """
        Huấn luyện mô hình.
        X: Ma trận đặc trưng (sparse matrix hoặc array), shape (n_samples, n_features)
        y: Nhãn (array), shape (n_samples,)
        """
        # Chuyển y thành array nếu chưa phải
        y = np.array(y)
        
        # Xác định các lớp (classes)
        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)
        n_features = X.shape[1]

        # Khởi tạo các biến đếm
        self.class_log_prior_ = np.zeros(n_classes)
        self.feature_log_prob_ = np.zeros((n_classes, n_features))

        # Tính toán cho từng lớp
        for idx, c in enumerate(self.classes_):
            # Lấy các mẫu thuộc lớp c
            X_c = X[y == c]
            
            # Tính xác suất tiên nghiệm (Prior) P(c)
            # Log probability = log(số mẫu lớp c / tổng số mẫu)
            self.class_log_prior_[idx] = np.log(X_c.shape[0] / X.shape[0])

            # Tính tổng số lần xuất hiện của từng từ trong lớp c
            # Cộng thêm alpha để làm smoothing (tránh xác suất = 0)
            count_word_in_class = X_c.sum(axis=0) + self.alpha
            
            # Tổng số từ trong toàn bộ lớp c (bao gồm cả alpha cho mỗi từ)
            total_count_in_class = count_word_in_class.sum()

            # Tính xác suất có điều kiện (Likelihood) P(x_i | c)
            # Log probability = log(số lần từ i xuất hiện trong c / tổng số từ trong c)
            self.feature_log_prob_[idx, :] = np.log(count_word_in_class / total_count_in_class)
            
        return self

    def predict_log_proba(self, X):
        """
        Tính log xác suất hậu nghiệm: log P(c | X) ~ log P(c) + sum(log P(x_i | c))
        """
        # X * feature_log_prob_.T:
        # (n_samples, n_features) x (n_features, n_classes) -> (n_samples, n_classes)
        # Đây là bước nhân ma trận để cộng tổng log likelihood của các từ trong câu
        jll = X @ self.feature_log_prob_.T + self.class_log_prior_
        return jll

    def predict_proba(self, X):
        """
        Chuyển đổi log proba sang xác suất thực (dùng hàm softmax hoặc chuẩn hóa).
        """
        jll = self.predict_log_proba(X)
        # Kỹ thuật log-sum-exp để tránh tràn số (overflow/underflow)
        # P(c|X) = exp(log P(c|X)) / sum(exp(log P(c'|X)))
        
        # Trừ max để ổn định số học
        jll_stable = jll - jll.max(axis=1, keepdims=True)
        exp_jll = np.exp(jll_stable)
        prob = exp_jll / exp_jll.sum(axis=1, keepdims=True)
        return prob

    def predict(self, X):
        """
        Dự đoán lớp có xác suất cao nhất.
        """
        jll = self.predict_log_proba(X)
        return self.classes_[np.argmax(jll, axis=1)]
    
    def score(self, X, y):
        """
        Tính độ chính xác (Accuracy).
        """
        y_pred = self.predict(X)
        return np.mean(y_pred == y)


# =========================================================
# 🔄 4️⃣ HÀM CROSS-VALIDATION TỰ VIẾT
# =========================================================
def custom_cross_val_score(model, X, y, cv=5):
    """
    Tự cài đặt K-Fold Cross Validation.
    """
    y = np.array(y)
    n_samples = X.shape[0]
    indices = np.arange(n_samples)
    
    # Xáo trộn dữ liệu (để đảm bảo ngẫu nhiên)
    np.random.seed(42)
    np.random.shuffle(indices)
    
    fold_sizes = np.full(cv, n_samples // cv, dtype=int)
    fold_sizes[:n_samples % cv] += 1
    
    current = 0
    scores = []
    
    print(f"   🔄 Running Custom {cv}-Fold CV...")
    
    for i in range(cv):
        start, stop = current, current + fold_sizes[i]
        test_indices = indices[start:stop]
        train_indices = np.concatenate([indices[:start], indices[stop:]])
        
        current = stop
        
        # Chia tập train/test
        X_train, X_test = X[train_indices], X[test_indices]
        y_train, y_test = y[train_indices], y[test_indices]
        
        # Clone model (tạo mới để không ảnh hưởng model gốc)
        # Ở đây ta khởi tạo mới đơn giản
        clf = CustomMultinomialNB(alpha=model.alpha)
        clf.fit(X_train, y_train)
        
        # Đánh giá
        score = clf.score(X_test, y_test)
        scores.append(score)
        
    return np.array(scores)


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
    # (1) Tùy chỉnh vectorizer để tối ưu cho Naive Bayes
    vectorizer.set_params(
        max_features=800,        # Giới hạn số đặc trưng để tránh ma trận quá thưa
        ngram_range=(1, 2),      # Học cả từ đơn (unigram) và cụm 2 từ (bigram)
        min_df=1                 # Giữ từ xuất hiện ít nhất 1 lần
    )

    # Biến đổi dữ liệu huấn luyện thành vector TF-IDF với cấu hình mới
    X_train = vectorizer.fit_transform(train_texts)

    # (2) Huấn luyện mô hình Custom Multinomial Naïve Bayes
    # alpha=0.1: Giảm smoothing để model "nhạy" hơn với các từ khóa đặc trưng
    nb_model = CustomMultinomialNB(alpha=0.1)
    nb_model.fit(X_train, train_labels)

    # (3) Đánh giá sơ bộ bằng custom cross-validation
    scores = custom_cross_val_score(nb_model, X_train, train_labels, cv=5)
    print(f"📊 Custom CV accuracy: {np.mean(scores):.3f} ± {np.std(scores):.3f}")

    # 💾 Lưu mô hình đã huấn luyện lại để dùng sau
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(nb_model, f)

    # Thông báo khi lưu thành công
    print("✅ Optimized Custom Naïve Bayes model saved at:", os.path.abspath(MODEL_PATH))

    return nb_model
