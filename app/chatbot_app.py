# -------------------------------
# 🧠 Chatbot học tập cho môn Nhập môn Trí tuệ Nhân tạo (IT3160)
# File này là "main.py" – file chính khởi chạy ứng dụng Flask
# 🔥 SO SÁNH 2 MÔ HÌNH: Naive Bayes vs KNN
# -------------------------------

# 📦 Import các thư viện cần thiết
from flask import Flask, render_template, request, redirect, url_for
import pickle
import os
import numpy as np
import sys

# ⚙️ CẤU HÌNH ĐƯỜNG DẪN
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

# Import các module xử lý NLP
from preprocess import preprocess_text, expand_query, detect_negation
from nb_module import predict_topic
from find_answer import find_best_answer
from knn_module import find_answer_knn  # 🆕 Import KNN
import pandas as pd

# CSV-based data loading (thay thế datastore.py)
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data')

def get_all_qa():
    """Load toàn bộ Q&A từ CSV"""
    df = pd.read_csv(os.path.join(DATA_DIR, 'qa_train.csv'))
    return df

def get_qa_by_topic(topic):
    """Lọc Q&A theo topic"""
    df = get_all_qa()
    return df[df['topic'] == topic]

# -------------------------------
# 🚀 Khởi tạo ứng dụng Flask
# -------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.join(BASE_DIR, "..")

app = Flask(
    __name__,
    template_folder=os.path.join(ROOT_DIR, "templates"),
    static_folder=os.path.join(ROOT_DIR, "static")
)

# -------------------------------
# 📂 Nạp mô hình học máy
# -------------------------------

# HACK: Fix lỗi Pickle load model cũ
if 'nb_module' not in sys.modules:
    from app import nb_module as pkg_nb_module
    sys.modules['nb_module'] = pkg_nb_module

# 🔥 Fix cho KNN module
if 'knn_module' not in sys.modules:
    from app import knn_module as pkg_knn_module
    sys.modules['knn_module'] = pkg_knn_module

# 1. Load Vectorizer
vectorizer = None
try:
    with open(os.path.join(BASE_DIR, '../models/vectorizer.pkl'), 'rb') as f:
        vectorizer = pickle.load(f)
    print("✅ Loaded TF-IDF Vectorizer.")
except Exception as e:
    print(f"⚠️ Could not load Vectorizer: {e}")

# 2. Load Naive Bayes model
nb_model = None
try:
    with open(os.path.join(BASE_DIR, '../models/nb_model.pkl'), 'rb') as f:
        nb_model = pickle.load(f)
    print("✅ Loaded Naive Bayes model.")
except Exception as e:
    print(f"⚠️ Could not load Naive Bayes model: {e}")

# 3. 🆕 Load KNN model
knn_model = None
try:
    with open(os.path.join(BASE_DIR, '../models/knn_model.pkl'), 'rb') as f:
        knn_model = pickle.load(f)
    print("✅ Loaded KNN model.")
except Exception as e:
    print(f"⚠️ Could not load KNN model: {e}")

# Biến toàn cục lưu lịch sử chat
chat_history = []

# =========================================================
# 🌐 ROUTES
# =========================================================
@app.route("/")
def home():
    return render_template("index.html", chat_history=chat_history)

@app.route("/get_response", methods=["POST"])
def chatbot():
    global chat_history
    
    if request.method == "POST":
        user_message = request.form.get('msg', '').strip()
        if not user_message:
            return redirect(url_for('chatbot'))

        # ---------------------------------------------------------
        # 🔍 BƯỚC 1: TIỀN XỬ LÝ
        # ---------------------------------------------------------
        expanded_query = expand_query(user_message)
        clean_query = detect_negation(preprocess_text(expanded_query))
        
        # ==========================================================
        # 🔍 KNN: Tìm câu trả lời trực tiếp
        # ==========================================================
        final_answer = None
        final_confidence = 0.0
        final_topic = "Unknown"
        matched_question = None
        
        if knn_model and vectorizer:
            try:
                # KNN tìm trực tiếp câu hỏi gần nhất
                answer, conf, matched_q, topic, _ = find_answer_knn(
                    knn_model, vectorizer, clean_query, k=3
                )
                final_answer = answer
                final_confidence = conf
                final_topic = topic if topic else "Unknown"
                matched_question = matched_q
                
            except Exception as e:
                print(f"❌ KNN Error: {e}")
        
        # ---------------------------------------------------------
        # 📊 LOGGING
        # ---------------------------------------------------------
        print(f"\n{'='*50}")
        print(f"📝 Query: {user_message}")
        print(f"🔍 [KNN] Topic: {final_topic} | Conf: {final_confidence:.2f}")
        print(f"{'='*50}\n")
        
        # ---------------------------------------------------------
        # 🤖 QUYẾT ĐỊNH TRẢ LỜI
        # ---------------------------------------------------------
        CONFIDENCE_THRESHOLD = 0.50
        
        if final_confidence < CONFIDENCE_THRESHOLD or not final_answer:
            final_answer = "Xin lỗi, tôi chưa hiểu câu hỏi của bạn. Bạn có thể diễn đạt lại không?"

        # Lưu lịch sử
        chat_history.append({
            "user": user_message,
            "bot": final_answer,
            "confidence": round(final_confidence, 2),
            "topic": final_topic,
        })
    
    return redirect(url_for('home'))

@app.route('/clear', methods=['POST'])
def clear_history():
    global chat_history
    chat_history = []
    return redirect(url_for('home'))

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5002)
