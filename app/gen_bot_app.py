from flask import Flask, render_template, request, redirect, url_for
import os
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# 1. Cấu hình
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.join(BASE_DIR, "..")
MODEL_PATH = os.path.join(ROOT_DIR, 'models', 'my_generative_bot')

app = Flask(
    __name__,
    template_folder=os.path.join(ROOT_DIR, "templates"),
    static_folder=os.path.join(ROOT_DIR, "static")
)

# 2. Load Model (Chỉ load 1 lần khi khởi động)
print("⏳ Đang tải model Generative Bot...")
try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_PATH)
    print("✅ Model đã sẵn sàng!")
except Exception as e:
    print(f"❌ Lỗi load model: {e}")
    print("⚠️ Hãy chắc chắn bạn đã chạy 'python app/train_generative.py' trước!")
    model = None

chat_history = []

def generate_answer(question):
    if not model:
        return "Lỗi: Model chưa được load."
    
    # Tiền xử lý input
    input_text = f"question: {question}"
    input_ids = tokenizer(input_text, return_tensors="pt").input_ids
    
    # Sinh câu trả lời
    outputs = model.generate(
        input_ids, 
        max_length=128, 
        num_beams=4, # Beam search cho kết quả mượt hơn
        early_stopping=True
    )
    
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

@app.route('/', methods=['GET', 'POST'])
def chatbot():
    global chat_history
    
    if request.method == 'POST':
        user_message = request.form['user_message']
        
        if user_message.strip():
            # Gọi model để sinh câu trả lời
            answer = generate_answer(user_message)
            
            chat_history.append({
                "user": user_message,
                "bot": answer,
                "confidence": 1.0,
                "topic": "Generative AI",
                "topic_conf": 1.0,      # Dummy value for template compatibility
                "question_sim": 1.0     # Dummy value for template compatibility
            })
        
        return redirect(url_for('chatbot'))
    
    return render_template('index.html', chat_history=chat_history)

@app.route('/clear', methods=['POST'])
def clear_history():
    global chat_history
    chat_history = []
    return redirect(url_for('chatbot'))

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001) # Chạy port 5001 vì port 5000 bị macOS chiếm dụng
