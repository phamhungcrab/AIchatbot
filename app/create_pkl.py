import pandas as pd
import pickle
import nltk
import os
import ssl

# Fix SSL for NLTK download
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

from datastore import get_all_qa
from preprocess import preprocess_text, train_vectorizer
from nb_module import train_naive_bayes

# Download NLTK data
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('punkt_tab', quiet=True)

BASE_DIR = os.path.dirname(__file__)
MODEL_DIR = os.path.join(BASE_DIR, '..', 'models')
# os.makedirs(MODEL_DIR, exist_ok=True)

def create_pkl_files():
    print('⏳ Đang tạo file .pkl...')
    
    # 1. Đọc dữ liệu
    df = get_all_qa()
    if df.empty:
        print('❌ Không có dữ liệu trong database!')
        return

    # 2. Tiền xử lý
    df['clean_text'] = df['question'].apply(preprocess_text)

    # 3. Tạo Vectorizer
    vectorizer = train_vectorizer(df['clean_text'])
    with open(os.path.join(MODEL_DIR, 'vectorizer.pkl'), 'wb') as f:
        pickle.dump(vectorizer, f)

    # 4. Tạo Model Naive Bayes
    nb_model = train_naive_bayes(vectorizer, df['clean_text'], df['topic'])
    
    # (KNN model đã bị bỏ, không cần tạo nữa)

    print('✅ Hoàn tất! Đã tạo vectorizer.pkl và nb_model.pkl trong thư mục models/.')

if __name__ == '__main__':
    create_pkl_files()
