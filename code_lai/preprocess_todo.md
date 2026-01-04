# 📝 TODO: Tự Code Lại `preprocess.py`

## 🎯 Mục tiêu
Code lại toàn bộ file `preprocess.py` từ đầu theo đúng cấu trúc hiện tại.

---

## Phase 1: Setup & Imports
- [ ] Tạo file `preprocess.py` mới (backup cái cũ)
- [ ] Import libraries:
  ```python
  import re
  from pyvi import ViTokenizer 
  from sklearn.feature_extraction.text import TfidfVectorizer
  ```

---

## Phase 2: Class TextPreprocessor - Khung cơ bản

### 2.1 Singleton Pattern
- [ ] Tạo class với `_instance = None`
- [ ] Override `__new__()` để chỉ tạo 1 object duy nhất
- [ ] Gọi `_initialize()` trong `__new__`

### 2.2 Method `_initialize()`
- [ ] Compile regex patterns:
  - [ ] `re_special_chars` - xóa ký tự đặc biệt
  - [ ] `re_numbers` - xóa số
- [ ] Gọi `_load_dictionaries()`
- [ ] Build synonym regex pattern

---

## Phase 3: Dictionaries - `_load_dictionaries()`

- [ ] **VIETNAMESE_STOPWORDS** (set) - 50+ từ
- [ ] **CRITICAL_KEYWORDS** (set) - thuật ngữ AI/ML cần giữ
- [ ] **LIGHT_STOPWORDS** (set) - stopwords nhẹ cho KNN
- [ ] **SYNONYMS** (dict) - từ đồng nghĩa
- [ ] **WEIGHTED_KEYWORDS** (dict) - từ khóa có trọng số
- [ ] **NEGATION_WORDS** (set) - từ phủ định
- [ ] **REVERSE_SYNONYMS** (dict) - mapping ngược

---

## Phase 4: Core Methods

### 4.1 `preprocess_text()` - Cho Naive Bayes
- [ ] Lowercase
- [ ] Xóa ký tự đặc biệt (regex)
- [ ] Xóa số (regex)
- [ ] Tokenize với PyVi
- [ ] Lọc stopwords (VIETNAMESE_STOPWORDS)
- [ ] Return string đã xử lý

### 4.2 `preprocess_for_knn()` - Cho KNN
- [ ] Lowercase + Clean (giữ số)
- [ ] Tokenize với PyVi
- [ ] Lọc với LIGHT_STOPWORDS
- [ ] Giữ CRITICAL_KEYWORDS
- [ ] Gọi `expand_query()`
- [ ] Return string đã mở rộng

---

## Phase 5: Helper Methods

- [ ] `expand_query()` - Thêm từ đồng nghĩa
- [ ] `detect_negation()` - Xử lý phủ định (NOT_token)
- [ ] `weighted_keyword_match()` - Tính điểm từ khóa
- [ ] `canonicalize_text()` - Chuẩn hóa về từ gốc
- [ ] `calculate_jaccard_similarity()` - Tính độ tương đồng

---

## Phase 6: Module-Level Interface

- [ ] Tạo singleton: `preprocessor = TextPreprocessor()`
- [ ] Expose wrapper functions:
  ```python
  def preprocess_text(text): return preprocessor.preprocess_text(text)
  def preprocess_for_knn(text): return preprocessor.preprocess_for_knn(text)
  # ... các hàm khác
  ```
- [ ] Viết `train_vectorizer(corpus)`

---

## Phase 7: Testing

- [ ] Viết sanity check `if __name__ == "__main__":`
- [ ] Test case 1: `"Học máy là gì?"` → NB output
- [ ] Test case 2: `"KNN khác gì Naive Bayes?"` → KNN output  
- [ ] So sánh output với file gốc

---

## ✅ Hoàn thành khi
- [ ] File chạy không lỗi
- [ ] Output giống file gốc
- [ ] Hiểu từng dòng code đã viết
