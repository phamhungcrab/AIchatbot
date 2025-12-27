# 📐 Giải thích Công thức Confidence

> Tài liệu hướng dẫn cho AI Chatbot  
> Ngày: 2025-12-23

---

## 🎯 Mục đích

**Confidence** = Mức độ "tự tin" của model khi đưa ra câu trả lời.

- 90% = "Tôi rất chắc!"
- 50% = "Có thể đúng, có thể sai"
- 20% = "Tôi đoán thôi..."

---

## 🤖 NAIVE BAYES

### Bước 1: Tính điểm cho mỗi Topic

Với câu hỏi có các từ: `["KNN", "là", "gì"]`

Model tính **điểm** cho từng topic:

| Topic | Công thức | Kết quả |
|-------|-----------|---------|
| Search | log(P_search) + log(P_knn\|search) + log(P_là\|search) + log(P_gì\|search) | -2.5 |
| **ML** | log(P_ml) + log(P_knn\|ml) + log(P_là\|ml) + log(P_gì\|ml) | **-0.3** ← cao nhất |
| Logic | log(P_logic) + log(P_knn\|logic) + ... | -3.1 |

### Bước 2: Chuyển điểm thành % (Softmax)

```
         e^(-0.3)           0.74
ML% = ─────────────── = ────────── = 74%
      e^(-2.5) + e^(-0.3) + e^(-3.1) + ...   1.0
```

### Bước 3: Lấy % cao nhất = Confidence

**Raw Confidence = 74%** (Topic: MachineLearning)

---

## 🌡️ TEMPERATURE SCALING

### Vấn đề

Model nói "74% chắc" nhưng thực tế chỉ đúng 47% → **quá tự tin!**

### Giải pháp: Chia cho Temperature

```
         e^(-0.3 / T)
ML% = ─────────────────
         Σ e^(score / T)
```

### Hiệu ứng của Temperature

| T | Kết quả | Mô tả |
|---|---------|-------|
| 0.5 | 85% | Rất tự tin |
| **1.0** | **65%** | **Bình thường** |
| 1.5 | 49% | Khiêm tốn |
| 2.0 | 40% | Rất khiêm tốn |

> **Công thức dễ nhớ:** T cao = confidence thấp, T thấp = confidence cao

---

## 🔍 KNN

### Bước 1: Tính Cosine Similarity

So sánh 2 vector TF-IDF:
- Vector A = câu hỏi user
- Vector B = câu hỏi trong database

```
                    A · B           (tích vô hướng)
Similarity = ───────────────── = ─────────────────────
             ||A|| × ||B||       (tích độ dài)
```

### Ví dụ trực quan

```
User: "KNN là gì"      → Vector [0.8, 0.1, 0.5, 0, 0, ...]
DB:   "KNN là thuật toán gì" → Vector [0.7, 0.2, 0.4, 0.1, 0, ...]

Similarity = 0.42 = 42%
```

### Bước 2: Sigmoid Scaling

Vấn đề: Similarity với TF-IDF thường chỉ 20-60%, nhìn thấp quá!

```
                        1
Calibrated = ─────────────────────────────
             1 + e^(-10 × (sim - 0.4))
```

| Raw Similarity | Sau Sigmoid |
|----------------|-------------|
| 0.2 | 12% |
| 0.3 | 27% |
| **0.4** | **50%** ← điểm giữa |
| 0.5 | 73% |
| 0.6 | 88% |

---

## 📊 TỔNG KẾT

### Công thức cuối cùng

| Model | Raw Confidence | Calibration | Calibrated |
|-------|----------------|-------------|------------|
| NB | max(softmax(scores)) | ÷ Temperature | softmax(scores/T) |
| KNN | cosine_similarity | Sigmoid | 1/(1+e^(-k(x-mid))) |

### Settings khuyến nghị

| Tham số | Giá trị | Lý do |
|---------|---------|-------|
| NB Temperature | 1.0 | Cân bằng tự tin/chính xác |
| KNN k | 10 | Sigmoid đủ dốc |
| KNN midpoint | 0.4 | Sim 40% = conf 50% |

---

## 🎓 TL;DR (Tóm tắt)

1. **NB tính điểm** cho mỗi topic → chuyển thành % → lấy % cao nhất
2. **Temperature** điều chỉnh mức tự tin (T cao = ít tự tin)
3. **KNN so sánh vector** → tính similarity → qua sigmoid cho đẹp
4. **Cả 2** đều output confidence 0-100%

---

*File này được tạo tự động bởi AI Chatbot system*
