# -------------------------------
# 🤖 knn_module.py — Mô-đun KNN cho Chatbot
# Chức năng: tìm câu trả lời gần nhất bằng độ tương đồng (cosine similarity)
# hoặc huấn luyện mô hình KNN để phân loại / truy hồi văn bản.
# -------------------------------

import numpy as np
# Dùng để tính độ tương đồng giữa 2 vector
from sklearn.metrics.pairwise import cosine_similarity
import pickle  # Dùng để lưu và nạp mô hình đã huấn luyện
import os

# -------------------------------
# 📁 Xác định đường dẫn đến file mô hình
# -------------------------------

# → thư mục hiện tại ("app/")
BASE_DIR = os.path.dirname(__file__)
# → lùi lên 1 cấp để đến thư mục "models"
MODEL_DIR = os.path.join(BASE_DIR, '..', 'models')
# os.makedirs(MODEL_DIR, exist_ok=True)                # (có thể mở lại nếu cần tự tạo thư mục models)
# → đường dẫn lưu file mô hình KNN
MODEL_PATH = os.path.join(MODEL_DIR, 'knn_model.pkl')

# =========================================================
# 🧭 1️⃣ HÀM TÌM CÂU TRẢ LỜI GẦN NHẤT (KNN hoặc Cosine)
# =========================================================


def find_best_answer(knn_model, vectorizer, question, df_topic):
    """
    ✅ Mục đích:
        Tìm câu trả lời gần nhất trong chủ đề hiện tại, dựa trên độ tương đồng giữa câu hỏi người dùng
        và các câu hỏi đã có trong cơ sở dữ liệu.

    📌 Tham số:
        knn_model: mô hình KNN (không bắt buộc, có thể chỉ dùng cosine_similarity)
        vectorizer: mô hình vector hóa (TF-IDF hoặc CountVectorizer)
        question: câu hỏi người dùng nhập vào (chuỗi string)
        df_topic: DataFrame chứa các câu hỏi và câu trả lời thuộc cùng một chủ đề
                   gồm 2 cột ['question', 'answer']

    🔁 Trả về:
        best_answer: câu trả lời phù hợp nhất (hoặc None nếu không có dữ liệu)
    """

    # ⚠️ Trường hợp không có dữ liệu trong chủ đề
    if df_topic.empty:
        return None

    # 🧩 Lấy danh sách các câu hỏi trong chủ đề
    corpus = df_topic['question'].tolist()

    # 🧮 Vector hóa tất cả câu hỏi trong chủ đề + câu hỏi mới của người dùng
    # → Biến văn bản thành vector số để tính toán được
    all_vectors = vectorizer.transform(corpus + [question])

    # 🔍 Tính độ tương đồng cosine giữa vector của người dùng và từng câu hỏi trong cơ sở tri thức
    # cosine_similarity cho biết 2 vector "giống nhau" bao nhiêu, giá trị từ 0 đến 1
    cosine_sim = cosine_similarity(all_vectors[-1], all_vectors[:-1])

    # 🥇 Lấy chỉ số (index) của câu hỏi có độ tương đồng cao nhất
    best_idx = np.argmax(cosine_sim)

    # 💬 Lấy câu trả lời tương ứng từ hàng có chỉ số đó trong DataFrame
    best_answer = df_topic.iloc[best_idx]['answer']

    # 🏁 Trả về câu trả lời cuối cùng
    return best_answer


# =========================================================
# 🧠 2️⃣ HÀM HUẤN LUYỆN MÔ HÌNH KNN
# =========================================================
def train_knn(vectorizer, train_texts, train_labels, n_neighbors):
    """
    ✅ Mục đích:
        Huấn luyện mô hình K-Nearest Neighbors (KNN) để tìm các câu hỏi tương tự
        hoặc phân loại dữ liệu văn bản (nếu có nhãn chủ đề).

    📌 Tham số:
        vectorizer: mô hình vector hóa TF-IDF (đã fit sẵn)
        train_texts: danh sách câu hỏi huấn luyện (list[str])
        train_labels: danh sách nhãn tương ứng (vd: chủ đề)
        n_neighbors: số lượng "láng giềng gần nhất" (k) cần xem xét

    🔁 Trả về:
        knn: mô hình KNN đã huấn luyện
    """
    # Import tại chỗ (bên trong hàm) để tránh load thư viện khi không cần
    from sklearn.neighbors import KNeighborsClassifier

    # ✳️ Chuyển danh sách văn bản sang dạng vector số
    X_train = vectorizer.transform(train_texts)

    # ⚙️ Khởi tạo mô hình KNN:
    # - metric='cosine' giúp đo độ tương đồng góc giữa các vector văn bản
    knn = KNeighborsClassifier(n_neighbors=n_neighbors, metric='cosine')

    # 🧩 Huấn luyện mô hình với dữ liệu huấn luyện
    knn.fit(X_train, train_labels)

    # 💾 Lưu mô hình đã huấn luyện xuống file .pkl để sử dụng lại sau này
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(knn, f)

    # 🏁 Trả về mô hình đã huấn luyện
    return knn
