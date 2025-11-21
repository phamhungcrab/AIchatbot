# -------------------------------
# 🤖 genai_module.py — Tích hợp Google Gemini
# -------------------------------

import os
import google.generativeai as genai
from dotenv import load_dotenv

# Nạp biến môi trường từ file .env
# Xác định đường dẫn tuyệt đối tới file .env (nằm ở thư mục gốc dự án)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ENV_PATH = os.path.join(BASE_DIR, '..', '.env')

print(f"DEBUG: Đang tìm file .env tại: {os.path.abspath(ENV_PATH)}")
load_dotenv(dotenv_path=ENV_PATH)

# Lấy API Key từ biến môi trường
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Debug: Kiểm tra xem key có load được không (chỉ in 5 ký tự đầu để bảo mật)
if GEMINI_API_KEY:
    print(f"DEBUG: Đã tìm thấy API Key: {GEMINI_API_KEY[:5]}...")
else:
    print("DEBUG: Không tìm thấy API Key trong biến môi trường.")

# Cấu hình Gemini
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
    # Sử dụng model Gemini 2.0 Flash (theo danh sách model của bạn)
    model = genai.GenerativeModel('gemini-2.0-flash')
else:
    model = None
    print("⚠️ CẢNH BÁO: Chưa cấu hình GEMINI_API_KEY trong file .env")

def generate_answer_with_ai(question, context_history=None):
    """
    Sinh câu trả lời bằng Google Gemini khi chatbot truyền thống bó tay.
    
    Args:
        question (str): Câu hỏi của người dùng.
        context_history (list): Lịch sử chat (tùy chọn) để AI hiểu ngữ cảnh.
        
    Returns:
        str: Câu trả lời từ AI.
    """
    if not model:
        return "Xin lỗi, chức năng AI chưa được cấu hình (thiếu API Key)."

    try:
        # Tạo prompt (lời nhắc) cho AI
        # Bạn có thể tùy chỉnh prompt này để AI đóng vai giảng viên/trợ giảng
        prompt = f"""
        Bạn là một trợ giảng nhiệt tình cho môn học "Nhập môn Trí tuệ Nhân tạo".
        Hãy trả lời câu hỏi sau của sinh viên một cách ngắn gọn, dễ hiểu và chính xác.
        Nếu câu hỏi không liên quan đến học tập hoặc AI, hãy từ chối khéo léo.
        
        Câu hỏi: {question}
        """
        
        # Gọi API để sinh nội dung
        response = model.generate_content(prompt)
        
        return response.text
    except Exception as e:
        print(f"❌ Lỗi khi gọi Gemini API: {e}")
        return "Xin lỗi, hiện tại tôi không thể kết nối với trí tuệ nhân tạo. Vui lòng thử lại sau."
