# -------------------------------
# 🧠 Chatbot học tập cho môn Nhập môn Trí tuệ Nhân tạo (IT3160)
# File này là "main.py" – file chính khởi chạy ứng dụng Flask
# -------------------------------

# -------------------------------
# 📦 Import các thư viện cần thiết
from flask import Flask, render_template, request, redirect, url_for  # Flask framework để xây web app
import pandas as pd              # Xử lý dữ liệu dạng bảng
import pickle                    # Đọc file model đã lưu (Naive Bayes, KNN, vectorizer)
import os                       # Thư viện thao tác với đường dẫn file/thư mục
import numpy as np

# Import TensorFlow cho Deep Learning (Safe Import)
try:
    import tensorflow as tf
    from tensorflow.keras.models import load_model
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    TF_AVAILABLE = True
except ImportError:
    print("⚠️ TensorFlow not found. Deep Learning features will be disabled.")
    TF_AVAILABLE = False

# Import các module xử lý NLP
from preprocess import preprocess_text, expand_query, detect_negation, weighted_keyword_match # 🆕 Module NLU đã gộp
from nb_module import predict_topic          # Hàm dự đoán chủ đề
from find_answer import find_best_answer      # Hàm tìm câu trả lời
from datastore import get_all_qa, get_qa_by_topic  # Các hàm truy xuất dữ liệu

# -------------------------------
# ⚙️ Thiết lập đường dẫn cho Flask
# -------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.join(BASE_DIR, "..")

# -------------------------------
# 🚀 Khởi tạo ứng dụng Flask
# -------------------------------
app = Flask(
    __name__,
    template_folder=os.path.join(ROOT_DIR, "templates"),
    static_folder=os.path.join(ROOT_DIR, "static")
)

# -------------------------------
# 🎛️ CẤU HÌNH MÔ HÌNH (MODEL CONFIGURATION)
# -------------------------------
# Chỉ bật DL nếu có thư viện TensorFlow VÀ người dùng muốn dùng
USE_DEEP_LEARNING = True if TF_AVAILABLE else False

# -------------------------------
# 📂 Nạp mô hình học máy đã huấn luyện sẵn
# -------------------------------

# 1. Naive Bayes & TF-IDF (Luôn nạp làm fallback)
try:
    with open('models/vectorizer.pkl', 'rb') as f:
        vectorizer = pickle.load(f)
    with open('models/nb_model.pkl', 'rb') as f:
        nb_model = pickle.load(f)
    print("✅ Loaded Naive Bayes model.")
except Exception as e:
    print(f"⚠️ Could not load Naive Bayes model: {e}")

# 2. Deep Learning (LSTM/GRU) - Chỉ nạp nếu cần hoặc file tồn tại
dl_model = None
dl_tokenizer = None
dl_label_encoder = None

if USE_DEEP_LEARNING:
    try:
        dl_model = load_model('models/dl_model.h5')
        with open('models/tokenizer.pickle', 'rb') as f:
            dl_tokenizer = pickle.load(f)
        with open('models/label_encoder.pickle', 'rb') as f:
            dl_label_encoder = pickle.load(f)
        print("✅ Loaded Deep Learning model.")
    except Exception as e:
        print(f"⚠️ Could not load Deep Learning model: {e}")
        print("➡️ Switching back to Naive Bayes.")
        USE_DEEP_LEARNING = False

# =========================================================
# 🧠 ENSEMBLE PREDICTION (Soft Voting)
# =========================================================
def predict_ensemble(text):
    """
    Kết hợp kết quả từ Naive Bayes và Deep Learning (nếu có).
    Chiến lược: Soft Voting (Trung bình cộng xác suất).
    """
    # 1. Dự đoán bằng Naive Bayes (Luôn khả dụng)
    nb_probs = None
    if nb_model and vectorizer:
        try:
            # Preprocess riêng cho NB
            expanded = expand_query(text)
            processed = preprocess_text(expanded)
            final_input = detect_negation(processed)

            X_nb = vectorizer.transform([final_input])
            nb_probs = nb_model.predict_proba(X_nb)[0]
            classes = nb_model.classes_
        except Exception as e:
            print(f"⚠️ NB Error: {e}")
            return "Unknown", 0.0

    # 2. Dự đoán bằng Deep Learning (Nếu khả dụng)
    dl_probs = None
    if USE_DEEP_LEARNING and dl_model and dl_tokenizer:
        try:
            # Preprocess riêng cho DL
            expanded = expand_query(text)
            processed = preprocess_text(expanded)
            final_input = detect_negation(processed)
            
            seq = dl_tokenizer.texts_to_sequences([final_input])
            padded = pad_sequences(seq, maxlen=100) # Max len khớp với lúc train
            
            dl_probs_raw = dl_model.predict(padded)[0]
            
            # Map DL probs sang đúng thứ tự classes của NB
            # (Giả sử LabelEncoder của DL khớp với classes của NB - Cần đồng bộ)
            # Để an toàn, ta dùng LabelEncoder của DL để map tên class -> prob
            dl_class_map = {dl_label_encoder.inverse_transform([i])[0]: p for i, p in enumerate(dl_probs_raw)}
            
            # Tạo vector prob theo thứ tự của NB classes
            dl_probs = np.zeros(len(classes))
            for i, cls in enumerate(classes):
                dl_probs[i] = dl_class_map.get(cls, 0.0)
                
        except Exception as e:
            print(f"⚠️ DL Error: {e}")
            dl_probs = None

    # 3. Kết hợp (Ensemble)
    if dl_probs is not None:
        # Trọng số: NB (0.4) + DL (0.6) - Ưu tiên DL vì hiểu ngữ cảnh tốt hơn
        final_probs = 0.4 * nb_probs + 0.6 * dl_probs
        print(f"🤖 Ensemble: NB({np.max(nb_probs):.2f}) + DL({np.max(dl_probs):.2f}) -> Final")
    else:
        # Fallback về NB 100%
        final_probs = nb_probs
        print(f"🤖 Ensemble: Only NB used ({np.max(nb_probs):.2f})")

    # 4. Lấy kết quả cuối cùng
    max_idx = np.argmax(final_probs)
    predicted_topic = classes[max_idx]
    confidence = final_probs[max_idx]
    
    return predicted_topic, confidence

# =========================================================
# 🌐 ROUTES
# =========================================================
@app.route("/")
def home():
    return render_template("index.html")

from inference_engine import inference_engine

@app.route("/get_response", methods=["POST"])
def chatbot():
                vectorizer, 
                final_input,  # ✅ Dùng input đã qua xử lý NLU
                df_topic, 
                original_query=user_message, # 🆕 Dùng query gốc cho Jaccard
                threshold=0.5
            )
            
            answer, question_similarity, matched_question = result
            
            # Bước 4: Tính điểm bổ sung từ từ khóa trọng số (Weighted Keywords)
            keyword_score = weighted_keyword_match(user_message) # Tính trên message gốc
            
            # ✅ Tính final confidence
            if answer is None:
                final_confidence = 0.0
            else:
                # Công thức mới có tính thêm keyword_score (nhẹ)
                base_conf = (0.60 * topic_confidence + 0.30 * question_similarity + 0.10 * 0.8)
                
                # Bonus điểm nếu khớp từ khóa quan trọng (tối đa +0.1)
                bonus = min(keyword_score * 0.05, 0.1)
                final_confidence = min(base_conf + bonus, 1.0)
                
                print(f"DEBUG: Base Conf={base_conf:.2f}, Bonus={bonus:.2f} -> Final={final_confidence:.2f}")

            # ---------------------------------------------------------
            # 🤖 QUYẾT ĐỊNH TRẢ LỜI (PURE NLU - NO GEN AI)
            # ---------------------------------------------------------
            CONFIDENCE_THRESHOLD = 0.55

            if final_confidence >= CONFIDENCE_THRESHOLD:
                # --- ĐỦ ĐỘ TIN CẬY ---
                if final_confidence >= 0.80:
                    pass
                elif final_confidence >= 0.65:
                    answer += "\n\n💡 (Tôi khá chắc chắn về câu trả lời này)"
                elif final_confidence >= 0.55:
                    answer += "\n\n⚠️ (Tôi không chắc lắm, bạn kiểm tra lại nhé)"
            else:
                # --- KHÔNG TÌM THẤY ---
                answer = "Xin lỗi, tôi chưa hiểu câu hỏi của bạn hoặc chưa được học về vấn đề này. Bạn hãy thử diễn đạt lại xem sao?"
                topic = "Unknown"
            
            # Lưu lịch sử
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
    global chat_history
    chat_history = []
    return redirect(url_for('chatbot'))


# -------------------------------
# ▶️ Chạy Flask app
# -------------------------------
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5002)
