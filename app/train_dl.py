import pandas as pd
import numpy as np
import pickle
import os
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder
from preprocess import preprocess_text
from dl_model import create_model

# Constants
MAX_NUM_WORDS = 5000
MAX_SEQUENCE_LENGTH = 100
EMBEDDING_DIM = 100
EPOCHS = 10
BATCH_SIZE = 32
MODEL_PATH = 'models/dl_model.h5'
TOKENIZER_PATH = 'models/tokenizer.pickle'
LABEL_ENCODER_PATH = 'models/label_encoder.pickle'

def train_dl_model():
    print("🚀 Starting Deep Learning Model Training...")

    # 1. Load Data
    if not os.path.exists('data/train_data.csv'):
        print("❌ Error: data/train_data.csv not found.")
        return

    df = pd.read_csv('data/train_data.csv')
    
    # 2. Preprocess Data
    print("🧹 Preprocessing text...")
    df['clean_text'] = df['question'].apply(preprocess_text)
    
    # 3. Tokenization & Padding
    print("🔠 Tokenizing...")
    tokenizer = Tokenizer(num_words=MAX_NUM_WORDS, filters='!"#$%&()*+,-./:;<=>?@[\\]^`{|}~\t\n', lower=True)
    tokenizer.fit_on_texts(df['clean_text'].values)
    word_index = tokenizer.word_index
    print(f"Found {len(word_index)} unique tokens.")

    X = tokenizer.texts_to_sequences(df['clean_text'].values)
    X = pad_sequences(X, maxlen=MAX_SEQUENCE_LENGTH)

    # 4. Encode Labels
    print("🏷️ Encoding labels...")
    # Assuming 'answer' is the target. In a real classification scenario, we might predict 'topic' or 'intent'.
    # Based on existing nb_module, it seems we predict 'topic'. Let's check train_data.csv structure.
    # If train_data.csv has 'topic' column, use it. If not, we might need to infer it or use 'answer' as class (which is bad if too many unique answers).
    # Let's assume we want to classify 'topic' like Naive Bayes does.
    # However, create_pkl.py suggests we might not have a 'topic' column in train_data.csv directly, 
    # but nb_module.py uses a 'topic' column.
    # Let's verify data columns first. If 'topic' is missing, we might need to generate it or use a different approach.
    # For now, I will assume 'topic' column exists or we are classifying unique answers (which acts as intents).
    # Let's check if 'topic' exists in df.
    
    if 'topic' not in df.columns:
        # Fallback: Treat each unique answer as a class (Intent Classification)
        # This works if answers are distinct per intent.
        target_column = 'answer'
    else:
        target_column = 'topic'
        
    le = LabelEncoder()
    Y = le.fit_transform(df[target_column])
    num_classes = len(le.classes_)
    print(f"Number of classes: {num_classes}")

    # 5. Build Model
    print("🏗️ Building model...")
    model = create_model(MAX_NUM_WORDS, EMBEDDING_DIM, MAX_SEQUENCE_LENGTH, num_classes)
    print(model.summary())

    # 6. Train Model
    print("🔥 Training...")
    model.fit(X, Y, epochs=EPOCHS, batch_size=BATCH_SIZE, validation_split=0.1, verbose=1)

    # 7. Save Artifacts
    print("💾 Saving artifacts...")
    if not os.path.exists('models'):
        os.makedirs('models')
        
    model.save(MODEL_PATH)
    
    with open(TOKENIZER_PATH, 'wb') as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
    with open(LABEL_ENCODER_PATH, 'wb') as handle:
        pickle.dump(le, handle, protocol=pickle.HIGHEST_PROTOCOL)

    print("✅ Training complete! Model saved to models/")

if __name__ == "__main__":
    train_dl_model()
