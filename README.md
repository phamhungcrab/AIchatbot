# AI Chatbot Project

**Tác giả:** Nguyễn Minh Khôi  
**MSSV:** 202416249

## 📖 Giới thiệu
Dự án AI Chatbot sử dụng kết hợp các thuật toán Machine Learning truyền thống (Naive Bayes, KNN) và Generative AI để trả lời câu hỏi của người dùng một cách chính xác và linh hoạt.

## ⚙️ Luồng hoạt động của Mô-đun Naive Bayes (`nb_module.py`)
Đây là thành phần nòng cốt giúp chatbot "hiểu" được chủ đề của câu hỏi. Quy trình xử lý diễn ra như sau:

### 1. Tiếp nhận & Vector hóa (Input & Vectorization)
*   **Đầu vào**: Câu hỏi dạng văn bản từ người dùng.
*   **Xử lý**: Văn bản được chuyển đổi thành các vector số học bằng kỹ thuật **TF-IDF** (Term Frequency-Inverse Document Frequency).
*   **Mục đích**: Giúp máy tính có thể tính toán và so sánh sự tương đồng giữa các câu.

### 2. Dự đoán chủ đề (Prediction)
*   Sử dụng thuật toán **Multinomial Naive Bayes** (được cài đặt tùy chỉnh trong lớp `CustomMultinomialNB`).
*   Mô hình sẽ tính toán xác suất câu hỏi thuộc về từng chủ đề đã biết (ví dụ: *Machine Learning, Python, Deep Learning, v.v.*).
*   Chủ đề có xác suất cao nhất sẽ được chọn làm dự đoán cuối cùng.

### 3. Đánh giá độ tin cậy (Confidence Score)
*   Hệ thống không chỉ đưa ra kết quả dự đoán mà còn kèm theo **độ tin cậy** (Confidence Score).
*   Độ tin cậy được tính dựa trên xác suất cao nhất tìm được.
*   **Quyết định luồng đi**:
    *   Nếu độ tin cậy **cao**: Chatbot trả lời ngay bằng dữ liệu có sẵn.
    *   Nếu độ tin cậy **thấp**: Chatbot sẽ chuyển câu hỏi sang mô hình **Generative AI (Gemini)** để xử lý tiếp.

### 4. Huấn luyện & Tối ưu (Training & Optimization)
*   Hệ thống hỗ trợ huấn luyện lại mô hình khi có dữ liệu mới.
*   Sử dụng kỹ thuật **K-Fold Cross-Validation** (tự cài đặt) để kiểm tra độ chính xác của mô hình trước khi lưu.
*   Mô hình tối ưu được lưu dưới dạng file `.pkl` để tái sử dụng.

---
## 🚀 Cài đặt & Chạy
1.  Cài đặt thư viện: `pip install -r requirements.txt`
2.  Chạy ứng dụng: `python3 app/chatbot_app.py`
