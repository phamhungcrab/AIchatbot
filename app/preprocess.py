# -------------------------------
# 🧹 preprocess.py — Mô-đun tiền xử lý văn bản
# Chức năng: làm sạch, chuẩn hóa dữ liệu ngôn ngữ tự nhiên
# để mô hình Naïve Bayes & KNN hiểu được.
# -------------------------------

import re                # Regular Expressions → xử lý ký tự đặc biệt, lọc chuỗi
import string            # Dùng để truy cập dấu câu (punctuation)
import nltk              # Natural Language Toolkit — thư viện xử lý ngôn ngữ tự nhiên
# Dùng để vector hóa văn bản (TF-IDF)
from sklearn.feature_extraction.text import TfidfVectorizer

# -------------------------------
# ⚙️ Thiết lập stopwords (từ dừng)
# -------------------------------
# Nếu là lần đầu chạy trên máy mới, bạn cần tải về dữ liệu stopwords:
# → Bỏ dấu "#" ở 2 dòng sau và chạy một lần
# nltk.download('stopwords')
# nltk.download('punkt')

from nltk.corpus import stopwords
# Tập hợp các từ dừng tiếng Anh (có thể mở rộng thêm tiếng Việt)
stop_words = set(stopwords.words('english'))


# =========================================================
# 🧠 1️⃣ HÀM TIỀN XỬ LÝ CHUỖI VĂN BẢN
# =========================================================
def preprocess_text(text: str) -> str:
    """
    ✅ Mục đích:
        Làm sạch và chuẩn hóa văn bản đầu vào để mô hình học máy xử lý tốt hơn.

    📌 Các bước thực hiện:
        1️⃣ Chuyển toàn bộ sang chữ thường.
        2️⃣ Loại bỏ ký tự đặc biệt, dấu câu, và số.
        3️⃣ Tách văn bản thành các từ riêng lẻ (tokenize).
        4️⃣ Xóa bỏ các từ dừng (stopwords) không mang nhiều ý nghĩa.
        5️⃣ Ghép lại thành chuỗi sạch cuối cùng.

    🔁 Trả về:
        clean_text: chuỗi văn bản đã được xử lý.
    """

    # 1️⃣ Chuyển tất cả ký tự về chữ thường để thống nhất
    text = text.lower()

    # 2️⃣ Loại bỏ ký tự đặc biệt, dấu chấm, dấu hỏi, v.v.
    # [^\w\s] nghĩa là giữ lại ký tự chữ và khoảng trắng, bỏ tất cả còn lại
    text = re.sub(r'[^\w\s]', '', text)

    # 3️⃣ Loại bỏ chữ số (số 0–9) để tránh nhiễu
    text = re.sub(r'\d+', '', text)

    # 4️⃣ Tokenization — tách văn bản thành danh sách các từ (tokens)
    tokens = nltk.word_tokenize(text)

    # 5️⃣ Loại bỏ stopwords (ví dụ: "the", "is", "are", "and"...)
    # → giúp mô hình tập trung vào từ khóa chính
    filtered_tokens = [word for word in tokens if word not in stop_words]

    # 6️⃣ Ghép lại danh sách từ thành chuỗi hoàn chỉnh (ngăn cách bằng khoảng trắng)
    clean_text = ' '.join(filtered_tokens)

    # 🏁 Trả về văn bản đã làm sạch
    return clean_text


# =========================================================
# 📊 2️⃣ HÀM HUẤN LUYỆN TF-IDF VECTORIZER
# =========================================================
def train_vectorizer(corpus):
    """
    ✅ Mục đích:
        Huấn luyện TF-IDF vectorizer từ danh sách văn bản (corpus).
        Sau đó có thể lưu vectorizer vào file .pkl để sử dụng lại.

    📌 Tham số:
        corpus: danh sách chuỗi văn bản (list[str]), ví dụ là các câu hỏi trong cơ sở dữ liệu

    🔁 Trả về:
        vectorizer: đối tượng TF-IDF đã được huấn luyện
    """

    # max_features=3000 → chỉ giữ lại 3000 từ quan trọng nhất (giúp giảm kích thước)
    vectorizer = TfidfVectorizer(max_features=3000)

    # "fit" để học ra bộ từ vựng và trọng số TF-IDF
    vectorizer.fit(corpus)

    # 🏁 Trả về mô hình vectorizer đã huấn luyện
    return vectorizer
