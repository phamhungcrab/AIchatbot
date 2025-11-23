# -------------------------------
# 🧹 preprocess.py — Tiền xử lý văn bản Tiếng Việt tối ưu
# -------------------------------

import re
import pickle
# Thư viện tách từ chuyên dụng cho tiếng Việt
from pyvi import ViTokenizer 
# Thư viện vector hóa văn bản
from sklearn.feature_extraction.text import TfidfVectorizer

# -------------------------------
# 🛑 1. DANH SÁCH STOPWORDS TIẾNG VIỆT (TỪ DỪNG)
# -------------------------------
# Đây là những từ xuất hiện nhiều nhưng ít mang ý nghĩa phân loại.
# Loại bỏ chúng giúp bot tập trung vào từ khóa chính (như "học máy", "giải thuật").
VIETNAMESE_STOPWORDS = {
    'thì', 'là', 'mà', 'và', 'của', 'những', 'các', 'như', 'thế', 'nào', 
    'được', 'về', 'với', 'trong', 'có', 'không', 'cho', 'tôi', 'bạn', 
    'cậu', 'tớ', 'mình', 'nó', 'hắn', 'gì', 'cái', 'con', 'người', 
    'sự', 'việc', 'đó', 'đây', 'kia', 'này', 'nhé', 'ạ', 'ơi', 'đi', 
    'làm', 'khi', 'lúc', 'nơi', 'tại', 'đã', 'đang', 'sẽ', 'muốn', 
    'phải', 'biết', 'hãy', 'rồi', 'chứ', 'nhỉ'
}

# =========================================================
# 🧠 2. HÀM TIỀN XỬ LÝ CHUỖI VĂN BẢN
# =========================================================
def preprocess_text(text: str) -> str:
    """
    Quy trình: Lowercase -> Xóa ký tự lạ -> Tách từ (PyVi) -> Lọc Stopwords
    """
    if not text:
        return ""

    # 1️⃣ Chuyển thành chữ thường
    text = text.lower()

    # 2️⃣ Xóa các ký tự đặc biệt (giữ lại chữ cái, số và dấu cách)
    # Loại bỏ dấu chấm, phẩy, hỏi chấm... để tránh nhiễu
    text = re.sub(r'[^\w\s]', '', text)
    
    # 3️⃣ Loại bỏ số (Tùy chọn: Nếu bot cần xử lý toán học thì bỏ dòng này)
    text = re.sub(r'\d+', '', text)

    # 4️⃣ Tách từ chuẩn tiếng Việt bằng PyVi
    # Quan trọng: "học máy" -> "học_máy", "trí tuệ nhân tạo" -> "trí_tuệ_nhân_tạo"
    # Giúp Bot hiểu đây là 1 cụm từ chứ không phải các từ rời rạc.
    tokenized_text = ViTokenizer.tokenize(text)

    # 5️⃣ Tách thành danh sách để lọc Stopwords
    tokens = tokenized_text.split()
    
    # 6️⃣ Lọc bỏ từ dừng và các từ quá ngắn (<= 1 ký tự)
    filtered_tokens = [
        word for word in tokens 
        if word not in VIETNAMESE_STOPWORDS and len(word) > 1
    ]

    # 7️⃣ Ghép lại thành chuỗi hoàn chỉnh
    return ' '.join(filtered_tokens)


# =========================================================
# 📊 3. HÀM HUẤN LUYỆN TF-IDF VECTORIZER (CÓ N-GRAM)
# =========================================================
def train_vectorizer(corpus):
    """
    Huấn luyện bộ chuyển đổi văn bản sang số (Vector).
    Cập nhật: Sử dụng N-gram để tăng độ tin cậy cho Naïve Bayes.
    """
    
    vectorizer = TfidfVectorizer(
        # ⭐️ Giảm xuống 800 để tối ưu cho Naive Bayes (tránh ma trận quá thưa)
        max_features=800,
        
        # ⭐️ QUAN TRỌNG: N-gram range (1, 2)
        # Giúp model học cả từ đơn ("học") và cụm 2 từ ("học_máy").
        # Điều này giúp tăng độ tin cậy (confidence score) lên rất nhiều.
        ngram_range=(1, 2),
        
        # Bỏ qua các từ xuất hiện quá ít (dưới 1 lần - mặc định)
        min_df=1,
        
        # ⭐️ Sublinear TF scaling: sử dụng log(tf) thay vì tf
        # Giúp giảm ảnh hưởng của từ xuất hiện quá nhiều lần
        sublinear_tf=True
    )

    # Học từ dữ liệu đầu vào
    vectorizer.fit(corpus)

    return vectorizer