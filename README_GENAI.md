# Hướng dẫn cài đặt AI (Google Gemini)

Để chatbot có thể tự sinh câu trả lời khi không tìm thấy trong dữ liệu, bạn cần làm theo các bước sau:

1.  **Lấy API Key miễn phí**:
    - Truy cập: [Google AI Studio](https://aistudio.google.com/app/apikey)
    - Đăng nhập và tạo một API Key mới.

2.  **Cấu hình biến môi trường**:
    - Tạo một file mới tên là `.env` trong thư mục gốc (cùng cấp với `app/`, `requirements.txt`).
    - Mở file `.env` và dán nội dung sau vào:
      ```
      GEMINI_API_KEY=paste_your_api_key_here
      ```
    - Thay `paste_your_api_key_here` bằng mã key bạn vừa lấy được.

3.  **Khởi động lại Chatbot**:
    - Tắt terminal đang chạy (Ctrl+C).
    - Chạy lại: `python3 app/chatbot_app.py`.

Chúc bạn thành công!
