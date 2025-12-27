# 📊 BÁO CÁO ĐÁNH GIÁ AI CHATBOT (CALIBRATED CONFIDENCE)

> **Thời gian đánh giá:** 2025-12-23 17:33:33  
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

### Confidence Calibration (Temperature Scaling)

| Tham số | Giá trị |
|---------|---------|
| **Method** | Temperature Scaling |
| **Formula** | `P_calibrated(c|X) = exp(log P(c|X) / T) / Σ exp(log P(k|X) / T)` |
| **Temperature** | 1.5 |
| **Ý nghĩa** | T > 1 làm mềm distribution → confidence thực tế hơn |

### Kết quả (RAW vs CALIBRATED)

| Metric | Raw | Calibrated |
|--------|-----|------------|
| **Accuracy** | 46.84% | (không đổi) |
| **Avg Confidence** | 64.58% | **48.57%** |
| **ECE** | 25.74% | **28.27%** |

### Coverage & Accuracy theo Threshold (Calibrated)

| Threshold | Coverage | Accuracy |
|-----------|----------|----------|
| ≥ 0.3 | 91.6% | 50.9% |
| ≥ 0.5 | 45.6% | 93.1% |
| ≥ 0.7 | 10.1% | 93.8% |
| ≥ 0.9 | 0.2% | 100.0% |

---

## 🔍 3. KNN (Tìm câu trả lời)

### Kỹ thuật sử dụng
- **Thuật toán:** Custom K-Nearest Neighbors
- **Distance Metric:** `Cosine Distance = 1 - (A·B)/(||A||×||B||)`
- **Raw Confidence:** `Raw Confidence = 1 - Cosine Distance = Cosine Similarity`
- **K neighbors:** 5

### Preprocessing cho KNN (Khác với NB)
- Lowercase
- Xóa ký tự đặc biệt (KHÔNG xóa số)
- PyVi Tokenizer (tách từ tiếng Việt)
- LIGHT_STOPWORDS (giữ từ khóa quan trọng)
- Synonym Expansion (mở rộng với từ đồng nghĩa)

### Confidence Calibration (Sigmoid Scaling)

| Tham số | Giá trị |
|---------|---------|
| **Method** | Sigmoid Scaling |
| **Formula** | `calibrated = 1 / (1 + exp(-k × (similarity - midpoint)))` |
| **k (steepness)** | 10.0 |
| **midpoint** | 0.4 |
| **Ý nghĩa** | Chuyển similarity từ [0.2-0.6] về [0-1] hợp lý hơn |

### Kết quả (RAW vs CALIBRATED)

| Metric | Raw | Calibrated |
|--------|-----|------------|
| **Exact Match** | 0.00% | (không đổi) |
| **Avg Confidence** | 42.84% | **50.45%** |

### Coverage & Accuracy theo Threshold (Calibrated)

| Threshold | Coverage | Accuracy |
|-----------|----------|----------|
| ≥ 0.3 | 54.9% | 0.0% |
| ≥ 0.5 | 51.5% | 0.0% |
| ≥ 0.7 | 25.9% | 0.0% |
| ≥ 0.9 | 21.3% | 0.0% |

---

## 📖 4. GIẢI THÍCH CALIBRATION

### Tại sao cần Calibration?
- **NB**: Confidence thường CAO quá (64% vs accuracy 47%) → Temperature Scaling làm "mềm"
- **KNN**: Cosine similarity thường THẤP với TF-IDF (0.2-0.6) → Sigmoid Scaling đưa về [0,1] hợp lý

### Metrics sau Calibration
| Metric | Ý nghĩa |
|--------|---------|
| **Accuracy** | Tỷ lệ % dự đoán đúng (không đổi) |
| **Avg Confidence** | Giá trị confidence trung bình ĐÃ CALIBRATE |
| **ECE** | Expected Calibration Error - lý tưởng nên ≈ 0% |
| **Coverage** | % mẫu có confidence ≥ threshold |

---

## 🧠 5. CÔNG THỨC CHI TIẾT

### NB Temperature Scaling
```
log_proba = log P(c) + Σ log P(word_i|c)
calibrated = softmax(log_proba / Temperature)
confidence = max(calibrated)
```

### KNN Sigmoid Scaling
```
raw_similarity = 1 - cosine_distance
calibrated = 1 / (1 + exp(-k × (raw_similarity - midpoint)))
```

---

*Báo cáo được tạo tự động bởi evaluate_models.py với calibrated confidence*
