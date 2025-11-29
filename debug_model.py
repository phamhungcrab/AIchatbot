from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import os
import torch

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'my_generative_bot')

print(f"📂 Đang tải model từ: {MODEL_PATH}")

try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_PATH)
    print("✅ Model loaded successfully!")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    exit()

def test_generate(question):
    print(f"\n❓ Câu hỏi: {question}")
    input_text = f"question: {question}"
    input_ids = tokenizer(input_text, return_tensors="pt").input_ids
    
    # Thử các tham số khác nhau
    print("--- Thử nghiệm 1 (Mặc định) ---")
    outputs = model.generate(input_ids, max_length=128, num_beams=4, early_stopping=True)
    print(f"Output: {tokenizer.decode(outputs[0], skip_special_tokens=True)}")

    print("--- Thử nghiệm 2 (Repetition Penalty) ---")
    outputs = model.generate(input_ids, max_length=128, num_beams=4, early_stopping=True, repetition_penalty=2.5)
    print(f"Output: {tokenizer.decode(outputs[0], skip_special_tokens=True)}")
    
    print("--- Thử nghiệm 4 (Aggressive Decoding) ---")
    outputs = model.generate(
        input_ids, 
        max_length=128, 
        num_beams=4, 
        repetition_penalty=3.0, 
        no_repeat_ngram_size=2,
        early_stopping=True
    )
    print(f"Output: {tokenizer.decode(outputs[0], skip_special_tokens=True)}")

test_generate("BFS là gì")
test_generate("DFS là gì")
