# 📊 BÁO CÁO ĐÁNH GIÁ AI CHATBOT

> **Thời gian đánh giá:** 2025-12-23 16:30:07  
> **Tổng số mẫu test:** 474

---

## 🧹 1. QUY TRÌNH TIỀN XỬ LÝ (Preprocessing)

- 1. Lowercase: Chuyển về chữ thường
- 2. Special Char Removal: Xóa ký tự đặc biệt bằng Regex r"[^\w\s]"
- 3. Number Removal: Xóa số bằng Regex r"\d+"
- 4. PyVi Tokenizer: Tách từ tiếng Việt (ViTokenizer.tokenize)
- 5. Stopword Removal: Lọc các từ dừng tiếng Việt (52 từ)

### TF-IDF Vectorization
| Tham số | Giá trị |
|---------|---------|
| max_features | 800 |
| ngram_range | (1, 2) - unigram + bigram |
| sublinear_tf | True |
| **Công thức** | `TF-IDF(t,d) = (1 + log(tf)) × log(N/df)` |

---

## 🤖 2. NAIVE BAYES (Phân loại Topic)

### Kỹ thuật sử dụng
- **Thuật toán:** Custom Multinomial Naive Bayes
- **Công thức:** `P(topic|X) ∝ P(topic) × ∏ P(word_i|topic)`
- **Smoothing:** Laplace Smoothing (alpha=0.1)

### Kết quả

| Metric | Giá trị |
|--------|---------|
| **Accuracy** | 46.84% |
| **Average Confidence** | 64.58% |
| **ECE (Calibration Error)** | 25.74% |

### Coverage & Accuracy theo Threshold

| Threshold | Coverage | Accuracy |
|-----------|----------|----------|
| ≥ 0.3 | 99.2% | 47.2% |
| ≥ 0.5 | 58.2% | 76.8% |
| ≥ 0.7 | 45.4% | 93.0% |
| ≥ 0.9 | 9.9% | 95.7% |

---

## 🔍 3. KNN (Tìm câu trả lời)

### Kỹ thuật sử dụng
- **Thuật toán:** Custom K-Nearest Neighbors
- **Distance Metric:** `Cosine Distance = 1 - (A·B)/(||A||×||B||)`
- **Confidence:** `Confidence = 1 - Cosine Distance = Cosine Similarity`
- **K neighbors:** 5

### Kết quả

| Metric | Giá trị |
|--------|---------|
| **Exact Match Accuracy** | 0.00% |
| **Average Confidence** | 43.67% |

### Coverage & Accuracy theo Threshold

| Threshold | Coverage | Accuracy |
|-----------|----------|----------|
| ≥ 0.3 | 54.4% | 0.0% |
| ≥ 0.5 | 30.8% | 0.0% |
| ≥ 0.7 | 8.9% | 0.0% |
| ≥ 0.9 | 2.3% | 0.0% |

---

## 📖 4. GIẢI THÍCH METRICS

| Metric | Ý nghĩa |
|--------|---------|
| **Accuracy** | Tỷ lệ % dự đoán đúng |
| **Average Confidence** | Giá trị confidence trung bình |
| **ECE** | Expected Calibration Error - độ "tin cậy" của confidence (càng thấp càng tốt) |
| **Coverage** | % mẫu có confidence ≥ threshold |
| **Accuracy@Threshold** | Accuracy tính trên các mẫu có confidence ≥ threshold |

---

## 🧠 5. TỔNG KẾT CÔNG NGHỆ

```
┌─────────────────────────────────────────────────────────────────┐
│                    AI CHATBOT ARCHITECTURE                      │
├─────────────────────────────────────────────────────────────────┤
│  INPUT: Câu hỏi người dùng                                      │
│    ↓                                                            │
│  PREPROCESSING: PyVi Tokenizer → Stopwords → TF-IDF             │
│    ↓                                                            │
│  ┌──────────────────────┐  ┌──────────────────────┐            │
│  │ NAIVE BAYES          │  │ KNN                  │            │
│  │ (Topic Classification)│  │ (Answer Retrieval)  │            │
│  │ • Multinomial NB     │  │ • Cosine Similarity  │            │
│  │ • P(C|X) ∝ P(C)ΠP(X|C)│ │ • k=5 neighbors      │            │
│  └──────────────────────┘  └──────────────────────┘            │
│    ↓                         ↓                                  │
│  OUTPUT: Topic + Confidence  OUTPUT: Answer + Confidence        │
└─────────────────────────────────────────────────────────────────┘
```

---

*Báo cáo được tạo tự động bởi evaluate_models.py*
