# -------------------------------
# 🧠 Chatbot học tập cho môn Nhập môn Trí tuệ Nhân tạo (IT3160)
# File này là "main.py" – file chính khởi chạy ứng dụng Flask
# -------------------------------

# 📦 Import các thư viện cần thiết
from flask import Flask, render_template, request, redirect, url_for  # Flask framework để xây web app
import pandas as pd              # Xử lý dữ liệu dạng bảng
import pickle                    # Đọc file model đã lưu (Naive Bayes, KNN, vectorizer)
from preprocess import preprocess_text       # Hàm tiền xử lý văn bản (loại bỏ stopword, ký tự đặc biệt...)
from nb_module import predict_topic          # Hàm dự đoán chủ đề bằng mô hình Naïve Bayes
from find_answer import find_best_answer      # Hàm tìm câu trả lời gần nhất bằng KNN
from datastore import get_all_qa, get_qa_by_topic  # Các hàm truy xuất dữ liệu Q&A từ SQLite

import os                       # Thư viện thao tác với đường dẫn file/thư mục
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

# -------------------------------
# ⚙️ Thiết lập đường dẫn cho Flask
# -------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.join(BASE_DIR, "..")

# -------------------------------
# 📂 Nạp mô hình Generative (Fallback)
# -------------------------------
MODEL_PATH = os.path.join(ROOT_DIR, 'models', 'my_generative_bot')
print("⏳ Đang tải model Generative Bot (Fallback)...")
try:
    gen_tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    gen_model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_PATH)
    print("✅ Generative Model đã sẵn sàng!")
except Exception as e:
    print(f"❌ Lỗi load Generative Model: {e}")
    gen_model = None

def generate_answer_local(question):
    if not gen_model:
        return None
    try:
        input_text = f"question: {question}"
        input_ids = gen_tokenizer(input_text, return_tensors="pt").input_ids
        outputs = gen_model.generate(input_ids, max_length=128, num_beams=4, early_stopping=True)
        return gen_tokenizer.decode(outputs[0], skip_special_tokens=True)
    except Exception as e:
        print(f"❌ Lỗi sinh câu trả lời: {e}")
        return None

# -------------------------------
# 🚀 Khởi tạo ứng dụng Flask
# -------------------------------
app = Flask(
    __name__,
    template_folder=os.path.join(ROOT_DIR, "templates"),  # Thư mục chứa file .html (Jinja2 templates)
    static_folder=os.path.join(ROOT_DIR, "static")        # Thư mục chứa CSS, JS, ảnh, favicon, v.v.
)

# -------------------------------
# 📂 Nạp mô hình học máy đã huấn luyện sẵn
# -------------------------------

# vectorizer.pkl: mô hình chuyển văn bản thành vector số (TF-IDF, Bag-of-Words, v.v.)
with open('models/vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

# nb_model.pkl: mô hình Naïve Bayes → dùng để dự đoán chủ đề (topic)
with open('models/nb_model.pkl', 'rb') as f:
    nb_model = pickle.load(f)

# knn_model.pkl: KHÔNG SỬ DỤNG (đã chuyển sang Cosine Similarity)
# with open('models/knn_model.pkl', 'rb') as f:
#     knn_model = pickle.load(f)

# -------------------------------
# 💬 Biến lưu lịch sử hội thoại
# -------------------------------
# Lưu tạm trong bộ nhớ RAM (dạng list), sẽ mất khi reload server
chat_history = []


# -------------------------------
# 🌐 ROUTE CHÍNH: Trang Chatbot
# -------------------------------
@app.route('/', methods=['GET', 'POST'])
def chatbot():
    global chat_history
    
    if request.method == 'POST':
        user_message = request.form['user_message']
        
        if user_message.strip():
            # Bước 1: Tiền xử lý
            processed = preprocess_text(user_message)
            
            # Bước 2: Dự đoán topic
            topic, topic_confidence = predict_topic(nb_model, vectorizer, processed)
            
            # Bước 3: Lấy câu hỏi trong topic
            df_topic = get_qa_by_topic(topic)
            
            # Bước 4: Tìm best match với threshold
            result = find_best_answer(
                vectorizer, 
                processed,  # ✅ Dùng processed thay vì user_message
                df_topic, 
                threshold=0.5  # ✅ Ngưỡng confidence tối thiểu
            )
            
            answer, question_similarity, matched_question = result
            
            # ✅ Tính final confidence
            # ✅ Tính final confidence
            if answer is None:
                # Case 1: Không tìm thấy câu hỏi nào trong DB (do threshold của find_best_answer)
                final_confidence = 0.0
                print("DEBUG: answer is None -> final_confidence = 0.0")
            else:
                # Case 2: Tìm thấy, nhưng cần kiểm tra độ tin cậy tổng hợp
                # ✅ Công thức mới (NB-Centric): Naive Bayes quyết định chính
                final_confidence = (
                    0.60 * topic_confidence +      # 60% - Naive Bayes quyết định chính
                    0.30 * question_similarity +   # 30% - Hỗ trợ tìm câu trả lời cụ thể
                    0.10 * 0.8                     # 10% - Yếu tố khác
                )
                print(f"DEBUG: Found answer. final_confidence = {final_confidence}")

            # ---------------------------------------------------------
            # 🤖 QUYẾT ĐỊNH: Dùng câu trả lời từ DB hay gọi AI?
            # ---------------------------------------------------------
            
            # Ngưỡng để chấp nhận câu trả lời từ DB (ví dụ: 0.55)
            CONFIDENCE_THRESHOLD = 0.55

            if final_confidence >= CONFIDENCE_THRESHOLD:
                # --- ĐỦ ĐỘ TIN CẬY ---
                print("DEBUG: Confidence >= Threshold. Using DB answer.")
                if final_confidence >= 0.80:
                    pass  # Rất tin cậy (>= 80%), không cần disclaimer
                elif final_confidence >= 0.65:
                    answer += "\n\n💡 Nếu câu trả lời chưa chính xác, hãy hỏi chi tiết hơn."
                elif final_confidence >= 0.55:
                    answer += "\n\n⚠️ Tôi không hoàn toàn chắc chắn. Bạn có thể hỏi theo cách khác?"
            else:
                # --- KHÔNG ĐỦ ĐỘ TIN CẬY (hoặc không tìm thấy) -> DÙNG GENERATIVE MODEL ---
                print(f"DEBUG: Confidence thấp ({final_confidence:.2f}). Chuyển sang Generative Model...")
                
                gen_answer = generate_answer_local(user_message)
                
                if gen_answer:
                    answer = gen_answer + "\n\n🤖 (Câu trả lời tự động từ AI)"
                    final_confidence = 0.9 # Giả định confidence cao cho AI
                    topic = "Generative AI"
                else:
                    answer = "Xin lỗi, tôi chưa được học về vấn đề này và AI cũng không trả lời được."
            
            # ✅ Lưu kèm confidence (optional - để debug/analysis)
            chat_history.append({
                "user": user_message,
                "bot": answer,
                "confidence": round(final_confidence, 3),
                "topic": topic,
                "topic_conf": round(topic_confidence, 3),
                "question_sim": round(question_similarity, 3) if question_similarity else 0.0
            })
        
        return redirect(url_for('chatbot'))
    
    return render_template('index.html', chat_history=chat_history)


# -------------------------------
# 🧹 ROUTE PHỤ: Xóa toàn bộ lịch sử chat
# -------------------------------
@app.route('/clear', methods=['POST'])
def clear_history():
    """
    Khi người dùng bấm nút 'Xóa lịch sử' → reset lại danh sách chat_history
    """
    global chat_history
    chat_history = []  # Làm trống danh sách hội thoại
    return redirect(url_for('chatbot'))  # Quay lại trang chatbot chính


# -------------------------------
# ▶️ Chạy Flask app
# -------------------------------
if __name__ == '__main__':
    # debug=True giúp auto reload khi thay đổi code và hiển thị log lỗi chi tiết
    app.run(debug=True, host='0.0.0.0', port=5002) # Chạy port 5002 để tránh AirPlay (port 5000)
