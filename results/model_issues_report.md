# 📋 BÁO CÁO LỖI NGẦM VÀ PHƯƠNG ÁN GIẢI QUYẾT

> **Ngày:** 2025-12-23  
> **Mục đích:** Phân tích các lỗi ngầm (hidden bugs) trong hệ thống AI Chatbot

---

## 🔴 DANH SÁCH LỖI NGẦM

### 1. ⚠️ Preprocessing quá aggressive cho KNN

**Vấn đề:**
- NB và KNN dùng chung `preprocess_text()` → xóa hết stopwords như `là`, `gì`, `khác`
- "KNN là gì" → chỉ còn `knn` → TF-IDF vector rất sparse
- Cosine similarity thấp dù câu hỏi đúng chủ đề

**Hậu quả:**
- Confidence của KNN luôn thấp với câu hỏi ngắn
- False negatives cao

**Giải pháp đã implement:**
```python
# preprocess.py
def preprocess_for_knn(text):
    # Dùng LIGHT_STOPWORDS thay vì VIETNAMESE_STOPWORDS
    # Giữ CRITICAL_KEYWORDS (thuật ngữ AI/ML)
    # Mở rộng với synonyms
```

| Trước | Sau |
|-------|-----|
| "KNN là gì" → `knn` | "KNN là gì" → `knn là gì k-nearest neighbors lân cận gần nhất` |

---

### 2. ⚠️ TF-IDF không capture semantic meaning

**Vấn đề:**
- TF-IDF chỉ so sánh từ → "học máy là gì" vs "machine learning là gì" → similarity thấp
- Không hiểu synonyms nếu không có trong training data

**Hậu quả:**
- Câu hỏi paraphrase khác nhau → không tìm được câu trả lời đúng

**Giải pháp:**
1. ✅ **Đã làm:** Synonym expansion trong `preprocess_for_knn()`
2. 🔮 **Upgrade tương lai:** Word embeddings (Word2Vec, FastText) hoặc Sentence-BERT

---

### 3. ⚠️ Confidence Calibration kém (ECE = 25.74%)

**Vấn đề:**
- Expected Calibration Error cao: model "tự tin" hơn khả năng thực tế
- Nếu confidence = 70%, accuracy thực tế chỉ ~50%

**Hậu quả:**
- User không thể tin tưởng vào confidence score
- Khó set threshold hợp lý

**Giải pháp:**
1. **Temperature Scaling:** Điều chỉnh logits bằng temperature parameter
   ```python
   calibrated_prob = softmax(logits / temperature)
   ```
2. **Platt Scaling:** Train logistic regression trên validation set

---

### 4. ⚠️ Không có fallback khi confidence thấp

**Vấn đề:**
- Khi confidence < threshold, chatbot vẫn trả về câu "gần nhất" (có thể sai)
- Không phân biệt được "không hiểu" vs "hiểu nhưng không chắc"

**Hậu quả:**
- Trả lời sai mà không cảnh báo user

**Giải pháp:**
```python
def get_answer(question, threshold=0.5):
    answer, confidence = knn_predict(question)
    
    if confidence >= 0.7:
        return answer  # Confident
    elif confidence >= threshold:
        return f"(Độ tin cậy: {confidence:.0%}) {answer}"  # Warning
    else:
        return "Xin lỗi, tôi không hiểu câu hỏi. Bạn có thể diễn đạt lại?"
```

---

### 5. ⚠️ Data Imbalance giữa các Topics

**Vấn đề:**
- Một số topic có nhiều Q&A, số khác rất ít
- NB có bias về topic phổ biến

**Check data:**
```python
df['topic'].value_counts()
# Ví dụ: MachineLearning: 500, Logic: 50 → imbalance 10:1
```

**Giải pháp:**
1. **Class weights:** Tăng weight cho topic thiểu số
   ```python
   from sklearn.utils.class_weight import compute_class_weight
   ```
2. **Oversampling:** Tạo thêm data cho topic ít
3. **Stratified sampling:** Đảm bảo validation set cân bằng

---

### 6. ⚠️ Training data và Validation data không cùng distribution

**Vấn đề:**
- Validation questions có thể được diễn đạt khác hoàn toàn với training
- KNN exact match = 0% (validation không có câu giống training)

**Hậu quả:**
- Metrics trên validation không reflect production performance

**Giải pháp:**
1. **Data augmentation:** Paraphrase training questions
2. **True semantic matching:** Dùng embedding thay vì exact TF-IDF match
3. **Fuzzy matching metric:** Thay exact match bằng semantic similarity score

---

### 7. ⚠️ MPS (Apple Silicon) chưa được tận dụng

**Vấn đề:**
- Code hiện tại chạy trên CPU
- Không tận dụng được Metal Performance Shaders trên M1/M2/M3

**Giải pháp (cho deep learning models):**
```python
import torch
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
model = model.to(device)
```

> 📌 Với NB/KNN từ sklearn, CPU đủ nhanh. MPS chỉ có ý nghĩa khi dùng PyTorch models.

---

## ✅ TỔNG KẾT

| Lỗi | Mức độ | Status |
|-----|--------|--------|
| Preprocessing cho KNN | 🔴 High | ✅ Đã fix |
| TF-IDF không semantic | 🟡 Medium | ⚠️ Cần upgrade |
| Confidence calibration | 🟡 Medium | 📝 Đề xuất |
| Không có fallback | 🔴 High | 📝 Đề xuất |
| Data imbalance | 🟡 Medium | 📝 Đề xuất |
| Train/Val mismatch | 🟡 Medium | 📝 Đề xuất |
| MPS acceleration | 🟢 Low | 📝 Optional |

---

*Báo cáo này được tạo tự động bởi evaluate_models.py*
