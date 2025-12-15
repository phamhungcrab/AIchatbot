import os
import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from preprocess import preprocess_text, expand_query, detect_negation

# Constants
MODEL_PATH = 'models/dl_model.h5'
TOKENIZER_PATH = 'models/tokenizer.pickle'
LABEL_ENCODER_PATH = 'models/label_encoder.pickle'
MAX_SEQUENCE_LENGTH = 100

def load_artifacts():
    print("📂 Loading artifacts...")
    if not os.path.exists(MODEL_PATH):
        print(f"❌ Error: Model file not found at {MODEL_PATH}")
        return None, None, None
        
    try:
        model = load_model(MODEL_PATH)
        with open(TOKENIZER_PATH, 'rb') as f:
            tokenizer = pickle.load(f)
        with open(LABEL_ENCODER_PATH, 'rb') as f:
            label_encoder = pickle.load(f)
        print("✅ Artifacts loaded successfully.")
        return model, tokenizer, label_encoder
    except Exception as e:
        print(f"❌ Error loading artifacts: {e}")
        return None, None, None

def predict_intent(text, model, tokenizer, label_encoder):
    # 1. NLU Preprocessing (Same as chatbot_app.py)
    expanded = expand_query(text)
    processed = preprocess_text(expanded)
    final_input = detect_negation(processed)
    
    print(f"\n🔍 Query: '{text}'")
    print(f"➡️ Processed: '{final_input}'")
    
    # 2. DL Preprocessing
    seq = tokenizer.texts_to_sequences([final_input])
    padded = pad_sequences(seq, maxlen=MAX_SEQUENCE_LENGTH)
    
    # 3. Predict
    pred_proba = model.predict(padded)
    max_idx = np.argmax(pred_proba)
    confidence = np.max(pred_proba)
    predicted_label = label_encoder.inverse_transform([max_idx])[0]
    
    return predicted_label, confidence

def main():
    model, tokenizer, label_encoder = load_artifacts()
    if not model:
        return

    # Test Cases
    test_queries = [
        "BFS là gì",
        "DFS khác BFS chỗ nào",
        "KNN dùng để làm gì",
        "Tác tử là gì",
        "Học máy có giám sát là sao"
    ]
    
    print("\n🚀 Running Test Cases...")
    for query in test_queries:
        label, conf = predict_intent(query, model, tokenizer, label_encoder)
        print(f"🤖 Prediction: [{label}] (Confidence: {conf:.4f})")
        print("-" * 30)

if __name__ == "__main__":
    main()
