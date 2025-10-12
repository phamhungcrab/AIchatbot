===============================================================
🎯 PROJECT: AICHATBOT — Chatbot học tập môn "Nhập môn Trí tuệ Nhân tạo"
===============================================================

📘 GIỚI THIỆU
---------------------------------------------------------------
Đây là chatbot web (Flask) giúp sinh viên hỏi–đáp về nội dung học phần IT3160 -
"Nhập môn Trí tuệ Nhân tạo" tại Đại học Bách khoa Hà Nội.

Chatbot hoạt động dựa trên:
- Mô hình Naïve Bayes: Dự đoán chủ đề của câu hỏi.
- Mô hình KNN + Cosine Similarity: Tìm câu hỏi tương tự nhất để trả lời.
- Dữ liệu huấn luyện lấy từ cơ sở dữ liệu SQLite (knowledge.db).
- Giao diện web sử dụng Flask + HTML (Jinja2) + CSS + JS.

---------------------------------------------------------------
📂 CẤU TRÚC THƯ MỤC DỰ ÁN
---------------------------------------------------------------

AICHATBOT/
│
├── app/                         ← Mã nguồn chính của Flask App
│   ├── __init__.py              ← Khởi tạo module Python
│   ├── chatbot_app.py           ← File Flask chính (chạy web server)
│   ├── datastore.py             ← Kết nối & truy vấn cơ sở dữ liệu SQLite
│   ├── preprocess.py            ← Xử lý văn bản (chuẩn hóa, xóa stopword,...)
│   ├── nb_module.py             ← Mô-đun huấn luyện & dự đoán bằng Naïve Bayes
│   ├── knn_module.py            ← Mô-đun tìm câu trả lời gần nhất bằng KNN
│   ├── train_models.py          ← Huấn luyện toàn bộ mô hình (TF-IDF, NB, KNN)
│   ├── testcode.py              ← Dùng để thử nghiệm nhanh mô hình (tuỳ chọn)
│   └── __pycache__/             ← Cache Python (tự sinh)
│
├── data/                        ← Thư mục chứa dữ liệu
│   ├── init.sql                 ← Câu lệnh SQL tạo bảng & nạp dữ liệu mẫu
│   ├── knowledge.db             ← Cơ sở dữ liệu SQLite (Q&A, topics,...)
│   └── seed_data.csv            ← File dữ liệu nguồn ban đầu (nếu có)
│
├── models/                      ← Nơi lưu các mô hình đã huấn luyện
│   ├── vectorizer.pkl           ← TF-IDF vectorizer
│   ├── nb_model.pkl             ← Mô hình Naïve Bayes
│   └── knn_model.pkl            ← Mô hình KNN
│
├── static/                      ← Tài nguyên giao diện web
│   ├── css/                     ← File CSS định dạng giao diện
│   ├── images/                  ← Ảnh favicon, logo HUST,...
│   └── js/                      ← File JavaScript (hiệu ứng chat, âm thanh,...)
│
├── templates/                   ← Các file giao diện HTML (Jinja2)
│   ├── base.html                ← Giao diện nền chung (header, nav, footer)
│   ├── index.html               ← Trang chính của chatbot
│   └── error.html               ← Trang hiển thị lỗi (nếu có)
│
├── venv/                        ← Môi trường ảo Python (tự sinh sau khi tạo)
│
├── requirements.txt             ← Danh sách thư viện Python cần cài
└── readme.txt                   ← File mô tả dự án (bạn đang đọc)


---------------------------------------------------------------
⚙️ CÀI ĐẶT VÀ CHẠY DỰ ÁN
---------------------------------------------------------------

1️⃣. Tạo môi trường ảo Python
---------------------------------------------------------------
python -m venv venv
venv\Scripts\activate.bat       (Windows)
source venv/bin/activate    (Linux/Mac)

2️⃣. Cài đặt thư viện cần thiết
---------------------------------------------------------------
pip install -r requirements.txt

3️⃣. Khởi tạo cơ sở dữ liệu (nếu chưa có)
---------------------------------------------------------------
cd app
python datastore.py
→ File knowledge.db sẽ được tạo trong thư mục /data

4️⃣. Huấn luyện mô hình
---------------------------------------------------------------
python app/train_models.py
→ Tạo các file model .pkl trong thư mục /models

5️⃣. Chạy web server Flask
---------------------------------------------------------------
python app/chatbot_app.py

→ Mở trình duyệt truy cập:
http://127.0.0.1:5000/
hoặc
http://localhost:5000/


---------------------------------------------------------------
💡 GHI CHÚ KỸ THUẬT
---------------------------------------------------------------
- Framework: Flask (Python)
- Machine Learning: scikit-learn (Naïve Bayes, KNN)
- Vector hóa: TF-IDF (TfidfVectorizer)
- Cơ sở dữ liệu: SQLite
- Frontend: HTML (Jinja2), CSS, JavaScript
- Môi trường: Python 3.12+

---------------------------------------------------------------
🧩 LUỒNG HOẠT ĐỘNG CHATBOT
---------------------------------------------------------------
1️⃣ Người dùng nhập câu hỏi → Flask nhận form (POST)
2️⃣ Văn bản được tiền xử lý (preprocess_text)
3️⃣ Naïve Bayes dự đoán chủ đề (predict_topic)
4️⃣ Lấy danh sách câu hỏi cùng chủ đề từ database
5️⃣ KNN / Cosine Similarity tìm câu hỏi giống nhất
6️⃣ Trả về câu trả lời tương ứng → hiển thị trên giao diện

---------------------------------------------------------------
bonus:
+ pip freeze > requirements.txt         (xuất thư viện vào requirements)

+ pip list --format=columns     (liệt kê thư viện)

+ where python      (check phiên bản python đang có)

+ where python
py --list   (kiểm tra các python đang có)

+ # Tạo 1 commit mới duy nhất

git checkout --orphan latest_branch (tạo lastest_branch mất lịch sử commit nhưng file code vẫn có)
git add -A (Thêm tất cả file hiện có (A = all) vào staging area.)
git commit -m "Initial clean commit" (Tạo commit đầu tiên (duy nhất) cho branch này.)

# Xóa branch cũ và đổi tên
git branch -D main (XÓA branch main cũ trên máy local (không phải GitHub)
git branch -m main (Đổi tên branch hiện tại (latest_branch) thành main.)

# Force push lên GitHub (ghi đè toàn bộ lịch sử)
git push -f origin main (Gửi branch main mới này lên GitHub và GHI ĐÈ lịch sử cũ)

+ https://www.python.org/downloads/release/python-3126/	(tải bản python 3.12)


---------------------------------------------------------------
👨‍💻 TÁC GIẢ
---------------------------------------------------------------
Phạm Ngọc Hưng — MSSV: 20235342
Trường Công nghệ Thông tin & Truyền thông
Đại học Bách khoa Hà Nội (HUST)
Môn học: IT3160 - Nhập môn Trí tuệ Nhân tạo
GVHD: Đỗ Tiến Dũng

===============================================================
📅 Ngày cập nhật: 08/10/2025
===============================================================
