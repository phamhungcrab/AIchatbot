# 📊 SO SÁNH NAIVE BAYES vs KNN

> **Dataset:** 474 mẫu validation  
> **Ngày test:** 2025-12-23

---

## 1. 🚀 TỐC ĐỘ

| Model | Thời gian (474 mẫu) | Tốc độ/câu |
|-------|---------------------|------------|
| **Naive Bayes** | 0.150 giây | **0.32 ms** |
| KNN | 0.416 giây | 0.88 ms |

> ✅ **Naive Bayes nhanh hơn 2.8 lần**

---

## 2. 📈 ĐỘ CHÍNH XÁC

| Metric | Naive Bayes | KNN |
|--------|-------------|-----|
| Topic Accuracy | **46.8%** | 45.1% |
| Avg Confidence | 64.6% | 50.5% |

> ✅ **Naive Bayes chính xác hơn 1.7%**

---

## 3. 🎯 ĐỘ TIN CẬY (Naive Bayes)

| Threshold | % câu đủ điều kiện | Accuracy |
|-----------|-------------------|----------|
| ≥ 50% | 58.2% | **76.8%** |
| ≥ 60% | 55.3% | **79.4%** |
| ≥ 80% | 38.2% | **94.5%** |

---

## ✅ KẾT LUẬN

| Tiêu chí | Naive Bayes | KNN | Thắng |
|----------|-------------|-----|-------|
| **Tốc độ** | 0.32 ms/câu | 0.88 ms/câu | 🏆 NB (2.8x) |
| **Accuracy** | 46.8% | 45.1% | 🏆 NB (+1.7%) |
| **Reliability** | 94.5%@80% | N/A | 🏆 NB |
| **Scalability** | O(1) predict | O(N) predict | 🏆 NB |

---

### Tóm tắt

**Naive Bayes tốt hơn vì:**
1. Nhanh hơn **2.8 lần**
2. Chính xác hơn **1.7%** 
3. Khi chỉ trả lời câu có conf ≥ 80%: đúng **94.5%**
4. Complexity O(1) tại prediction time

**KNN vẫn hữu dụng cho:**
- Tìm câu trả lời cụ thể (retrieval)
- Không cần train lại khi thêm data
