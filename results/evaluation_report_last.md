# 📊 BÁO CÁO ĐÁNH GIÁ AI CHATBOT - FINAL

> **Thời gian đánh giá:** 2025-12-28  
> **Tổng mẫu test:** 200  
> **Tổng mẫu train:** 1000

---

## 1. TỔNG QUAN KẾT QUẢ

| Model | Metric | Giá trị |
|-------|--------|---------|
| **Naive Bayes** | Topic Classification Accuracy | **88.50%** |
| **Naive Bayes** | Answer Accuracy (full pipeline) | **49.50%** |
| **KNN** | Exact Match Accuracy (Top-1) | **56.50%** |
| **KNN** | Weighted Voting Accuracy | 56.00% |

---

## 2. NAIVE BAYES

### 2.1. Kỹ thuật
- **Thuật toán:** Custom Multinomial Naive Bayes
- **Công thức:** P(topic|X) ∝ P(topic) × ∏ P(word_i|topic)
- **Smoothing:** Laplace Smoothing (α = 0.1)

### 2.2. Kết quả Chi tiết

| Metric | Raw | Calibrated |
|--------|-----|------------|
| Topic Accuracy | 88.50% | 88.50% |
| **Answer Accuracy** | **49.50%** | **49.50%** |
| Avg Confidence | 79.74% | 66.27% |

### 2.3. Accuracy theo Threshold (Calibrated)

| Threshold | Accuracy | Coverage |
|-----------|----------|----------|
| ≥ 30% | 96.5% | 86.5% |
| ≥ 50% | 99.3% | 74.0% |
| ≥ 70% | 100% | 55.0% |
| ≥ 90% | 100% | 18.0% |

---

## 3. KNN

### 3.1. Kỹ thuật
- **Thuật toán:** Custom K-Nearest Neighbors
- **Distance Metric:** Cosine Distance = 1 - Cosine Similarity
- **K neighbors:** 5

### 3.2. So sánh Phương pháp

| Phương pháp | Accuracy |
|-------------|----------|
| Top-1 (Nearest) | **56.50%** |
| Weighted Voting | 56.00% |

→ Top-1 tốt hơn 0.5%, đơn giản hơn

### 3.3. Accuracy theo Threshold

| Threshold | Accuracy | Coverage |
|-----------|----------|----------|
| ≥ 50% | 68.2% | 78.5% |
| ≥ 70% | 90.2% | 56.0% |
| ≥ 90% | 93.8% | 32.0% |

---

## 4. SO SÁNH TỔNG HỢP

| Tiêu chí | Naive Bayes | KNN |
|----------|-------------|-----|
| **Topic Classification** | ✅ 88.5% | - |
| **Answer Accuracy** | 49.5% | ✅ **56.5%** |
| Độ phức tạp | 2 bước (NB + find_answer) | 1 bước |

### Nhận xét

1. **NB Pipeline (NB + find_answer):** Accuracy = 49.5%
   - NB phân loại topic rất tốt (88.5%)
   - Nhưng find_answer trong topic đó chỉ đạt ~56% → tổng pipeline = 49.5%

2. **KNN Direct:** Accuracy = 56.5%
   - Tìm trực tiếp câu hỏi gần nhất trong toàn bộ database
   - Đơn giản hơn và hiệu quả hơn 7%

3. **Khuyến nghị:**
   - Sử dụng **KNN Direct** cho answer retrieval
   - Sử dụng **NB** nếu cần biết topic của câu hỏi
   - Áp dụng ngưỡng confidence ≥ 70% để đạt accuracy > 90%

---

## 5. CONFIDENCE CALIBRATION

### Naive Bayes - Temperature Scaling
```
P_calibrated(c|X) = exp(log P(c|X) / T) / Σ exp(log P(k|X) / T)
Temperature = 1.5
```

### KNN - Sigmoid Scaling
```
calibrated = 1 / (1 + exp(-k × (similarity - midpoint)))
k = 10.0, midpoint = 0.4
```

---

*Báo cáo được tạo tự động bởi evaluate_models.py*
