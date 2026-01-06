# -------------------------------
# 🧠 AI Chatbot - Streamlit Version
# Môn: Nhập môn Trí tuệ Nhân tạo (IT3160)
# Sử dụng KNN để tìm câu trả lời
# -------------------------------

import streamlit as st
import pickle
import os
import sys
import pandas as pd

# ⚙️ CẤU HÌNH ĐƯỜNG DẪN
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
APP_DIR = os.path.join(BASE_DIR, 'app')
MODEL_DIR = os.path.join(BASE_DIR, 'models')
DATA_DIR = os.path.join(BASE_DIR, 'data')

# Thêm app vào path để import modules
if APP_DIR not in sys.path:
    sys.path.insert(0, APP_DIR)
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

# Import các module xử lý NLP
from preprocess import preprocess_text, expand_query, detect_negation
from knn_module import find_answer_knn

# HACK: Fix lỗi Pickle load model cũ
if 'knn_module' not in sys.modules:
    from app import knn_module as pkg_knn_module
    sys.modules['knn_module'] = pkg_knn_module

# -------------------------------
# 🎨 CẤU HÌNH GIAO DIỆN
# -------------------------------
st.set_page_config(
    page_title="🤖 AI Chatbot - IT3160",
    page_icon="🧠",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Custom CSS
st.markdown("""
<style>
    /* Main container */
    .stApp {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f0f23 100%);
    }

    /* Title styling */
    h1 {
        background: linear-gradient(90deg, #00d4ff, #7b2cbf);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 700 !important;
        text-align: center;
    }

    /* Chat message styling */
    .chat-message {
        padding: 1rem;
        border-radius: 12px;
        margin-bottom: 0.75rem;
        animation: fadeIn 0.3s ease-in;
    }

    .user-message {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        margin-left: 15%;
    }

    .bot-message {
        background: linear-gradient(135deg, #2d3748 0%, #1a202c 100%);
        color: #e2e8f0;
        margin-right: 15%;
        border: 1px solid #4a5568;
    }

    /* Confidence badge */
    .confidence-badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 600;
        margin-top: 0.5rem;
    }

    .conf-high { background: #48bb78; color: white; }
    .conf-medium { background: #ecc94b; color: #1a202c; }
    .conf-low { background: #f56565; color: white; }

    /* Topic badge */
    .topic-badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        font-size: 0.75rem;
        background: #4299e1;
        color: white;
        margin-left: 0.5rem;
    }

    /* Input styling */
    .stTextInput > div > div > input {
        background: #2d3748 !important;
        border: 2px solid #4a5568 !important;
        border-radius: 12px !important;
        color: white !important;
        padding: 0.75rem 1rem !important;
    }

    .stTextInput > div > div > input:focus {
        border-color: #667eea !important;
        box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.3) !important;
    }

    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 0.75rem 2rem !important;
        font-weight: 600 !important;
        transition: all 0.3s ease !important;
    }

    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 4px 20px rgba(102, 126, 234, 0.4) !important;
    }

    /* Sidebar styling */
    .css-1d391kg {
        background: #1a202c;
    }

    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
</style>
""", unsafe_allow_html=True)


# -------------------------------
# 📂 LOAD MODELS (cached)
# -------------------------------
@st.cache_resource
def load_models():
    """Load TF-IDF Vectorizer và KNN Model (chỉ load 1 lần)"""
    vectorizer = None
    knn_model = None

    # Load Vectorizer
    vectorizer_path = os.path.join(MODEL_DIR, 'vectorizer.pkl')
    if os.path.exists(vectorizer_path):
        with open(vectorizer_path, 'rb') as f:
            vectorizer = pickle.load(f)
        print("✅ Loaded TF-IDF Vectorizer")

    # Load KNN Model
    knn_path = os.path.join(MODEL_DIR, 'knn_model.pkl')
    if os.path.exists(knn_path):
        with open(knn_path, 'rb') as f:
            knn_model = pickle.load(f)
        print("✅ Loaded KNN Model")

    return vectorizer, knn_model


# -------------------------------
# 🤖 HÀM XỬ LÝ CHATBOT
# -------------------------------
def get_chatbot_response(user_message, vectorizer, knn_model):
    """
    Xử lý câu hỏi và trả về câu trả lời.

    Returns:
        (answer, confidence, topic, matched_question)
    """
    if not vectorizer or not knn_model:
        return "⚠️ Model chưa được load. Vui lòng kiểm tra thư mục models/", 0.0, "Error", None

    # 1. Tiền xử lý
    expanded_query = expand_query(user_message)
    clean_query = detect_negation(preprocess_text(expanded_query))

    # 2. Tìm câu trả lời bằng KNN
    try:
        answer, confidence, matched_q, topic, _ = find_answer_knn(
            knn_model, vectorizer, clean_query, k=3
        )

        # 3. Kiểm tra ngưỡng tin cậy
        CONFIDENCE_THRESHOLD = 0.50
        if confidence < CONFIDENCE_THRESHOLD or not answer:
            answer = "Xin lỗi, tôi chưa hiểu câu hỏi của bạn. Bạn có thể diễn đạt lại không? 🤔"

        return answer, confidence, topic if topic else "Unknown", matched_q

    except Exception as e:
        return f"❌ Lỗi: {str(e)}", 0.0, "Error", None


def get_confidence_class(confidence):
    """Trả về CSS class dựa trên độ tin cậy"""
    if confidence >= 0.7:
        return "conf-high"
    elif confidence >= 0.5:
        return "conf-medium"
    else:
        return "conf-low"


# -------------------------------
# 🚀 MAIN APP
# -------------------------------
def main():
    # Load models
    vectorizer, knn_model = load_models()

    # Header
    st.markdown("# 🧠 AI Chatbot")
    st.markdown("<p style='text-align: center; color: #a0aec0;'>Trợ lý học tập môn Nhập môn Trí tuệ Nhân tạo (IT3160)</p>", unsafe_allow_html=True)

    # Model status
    if vectorizer and knn_model:
        st.success("✅ Models loaded successfully!")
    else:
        st.error("❌ Không thể load models. Kiểm tra thư mục `models/`")
        st.stop()

    # Divider
    st.markdown("---")

    # Initialize chat history in session state
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat history
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            st.markdown(f"""
                <div class="chat-message user-message">
                    <strong>🧑 Bạn:</strong><br>{msg["content"]}
                </div>
            """, unsafe_allow_html=True)
        else:
            conf = msg.get("confidence", 0)
            topic = msg.get("topic", "Unknown")
            conf_class = get_confidence_class(conf)

            st.markdown(f"""
                <div class="chat-message bot-message">
                    <strong>🤖 Bot:</strong><br>{msg["content"]}
                    <br>
                    <span class="confidence-badge {conf_class}">
                        📊 Độ tin cậy: {conf:.0%}
                    </span>
                    <span class="topic-badge">📚 {topic}</span>
                </div>
            """, unsafe_allow_html=True)

    # Chat input
    st.markdown("<br>", unsafe_allow_html=True)

    # Use columns for input and button layout
    col1, col2 = st.columns([5, 1])

    with col1:
        user_input = st.text_input(
            "Nhập câu hỏi của bạn:",
            placeholder="Ví dụ: BFS là gì?",
            key="user_input",
            label_visibility="collapsed"
        )

    with col2:
        send_button = st.button("Gửi 🚀", use_container_width=True)

    # Process input
    if send_button and user_input:
        # Add user message
        st.session_state.messages.append({
            "role": "user",
            "content": user_input
        })

        # Get bot response
        with st.spinner("🔍 Đang tìm câu trả lời..."):
            answer, confidence, topic, matched_q = get_chatbot_response(
                user_input, vectorizer, knn_model
            )

        # Add bot response
        st.session_state.messages.append({
            "role": "assistant",
            "content": answer,
            "confidence": confidence,
            "topic": topic,
            "matched_question": matched_q
        })

        # Rerun to update UI
        st.rerun()

    # Sidebar with extra info
    with st.sidebar:
        st.markdown("## ⚙️ Thông tin")
        st.markdown(f"""
        - **Model:** KNN with Cosine Similarity
        - **Vectorizer:** TF-IDF
        - **Threshold:** 50%
        """)

        st.markdown("---")

        # Clear chat button
        if st.button("🗑️ Xóa lịch sử chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

        st.markdown("---")
        st.markdown("### 📝 Ví dụ câu hỏi:")
        example_questions = [
            "BFS là gì?",
            "DFS khác BFS như thế nào?",
            "KNN hoạt động ra sao?",
            "Thuật toán A* là gì?",
        ]
        for q in example_questions:
            if st.button(q, key=f"example_{q}", use_container_width=True):
                # Set as input
                st.session_state.messages.append({"role": "user", "content": q})
                answer, confidence, topic, matched_q = get_chatbot_response(
                    q, vectorizer, knn_model
                )
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "confidence": confidence,
                    "topic": topic,
                    "matched_question": matched_q
                })
                st.rerun()


if __name__ == "__main__":
    main()
