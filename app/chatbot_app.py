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
        # 🧠 MODEL 1: NAIVE BAYES + COSINE/JACCARD
        # ==========================================================
        nb_answer = None
        nb_confidence = 0.0
        nb_topic = "Unknown"
        nb_topic_conf = 0.0
        nb_matched_q = None
        
        if nb_model and vectorizer:
            try:
                # Predict Topic
                nb_topic, nb_topic_conf = predict_topic(nb_model, vectorizer, clean_query)
                
                # Get data for topic
                df_topic = get_qa_by_topic(nb_topic)
                if df_topic.empty or nb_topic_conf < 0.4:
                    df_topic = get_all_qa()
                
                # Find answer using Cosine + Jaccard
                if not df_topic.empty:
                    ans, sim_score, matched = find_best_answer(
                        vectorizer, clean_query, df_topic,
                        original_query=user_message, threshold=0.3
                    )
                    nb_answer = ans
                    nb_matched_q = matched
                    
                    # Calculate final confidence
                    if sim_score and sim_score > 0:
                        nb_confidence = 0.7 * nb_topic_conf + 0.3 * sim_score
                        if nb_topic_conf >= 0.9:
                            nb_confidence = max(nb_confidence, 0.85)
                        elif nb_topic_conf >= 0.7:
                            nb_confidence = max(nb_confidence, 0.80)
                            
            except Exception as e:
                print(f"❌ NB Error: {e}")
        
        # ==========================================================
        # 🔍 MODEL 2: KNN (DIRECT SEARCH)
        # ==========================================================
        knn_answer = None
        knn_confidence = 0.0
        knn_topic = "Unknown"
        knn_matched_q = None
        
        if knn_model and vectorizer:
            try:
                # KNN tìm trực tiếp câu hỏi gần nhất (không qua topic)
                answer, conf, matched_q, topic, _ = find_answer_knn(
                    knn_model, vectorizer, clean_query, k=3
                )
                knn_answer = answer
                knn_confidence = conf
                knn_topic = topic if topic else "Unknown"
                knn_matched_q = matched_q
                
            except Exception as e:
                print(f"❌ KNN Error: {e}")
        
        # ---------------------------------------------------------
        # 📊 LOGGING SO SÁNH
        # ---------------------------------------------------------
        print(f"\n{'='*50}")
        print(f"📝 Query: {user_message}")
        print(f"🧠 [NB]  Topic: {nb_topic} ({nb_topic_conf:.2f}) | Conf: {nb_confidence:.2f}")
        print(f"🔍 [KNN] Topic: {knn_topic} | Conf: {knn_confidence:.2f}")
        print(f"{'='*50}\n")
        
        # ---------------------------------------------------------
        # 🤖 QUYẾT ĐỊNH TRẢ LỜI (Chọn model tốt hơn)
        # ---------------------------------------------------------
        CONFIDENCE_THRESHOLD = 0.60  # Hạ ngưỡng để so sánh được
        
        # Chọn model có confidence cao hơn
        if nb_confidence >= knn_confidence:
            winner = "Naive Bayes"
            final_answer = nb_answer
            final_confidence = nb_confidence
            final_topic = nb_topic
        else:
            winner = "KNN"
            final_answer = knn_answer
            final_confidence = knn_confidence
            final_topic = knn_topic
        
        # Kiểm tra ngưỡng
        if final_confidence < CONFIDENCE_THRESHOLD or not final_answer:
            final_answer = "Xin lỗi, tôi chưa hiểu câu hỏi của bạn. Bạn có thể diễn đạt lại không?"
            winner = "None"

        # Lưu lịch sử với thông tin so sánh
        chat_history.append({
            "user": user_message,
            "bot": final_answer,
            "confidence": round(final_confidence, 2),
            "topic": final_topic,
            "topic_conf": round(nb_topic_conf, 2),
            # 🆕 So sánh 2 model
            "nb_conf": round(nb_confidence, 2),
            "knn_conf": round(knn_confidence, 2),
            "winner": winner,
            "nb_answer": nb_answer[:100] + "..." if nb_answer and len(nb_answer) > 100 else nb_answer,
            "knn_answer": knn_answer[:100] + "..." if knn_answer and len(knn_answer) > 100 else knn_answer,
        })
    
    return redirect(url_for('home'))

@app.route('/clear', methods=['POST'])
def clear_history():
    global chat_history
    chat_history = []
    return redirect(url_for('home'))

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5002)
