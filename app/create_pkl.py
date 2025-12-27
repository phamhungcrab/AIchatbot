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

# CSV-based data loading (thay thế datastore.py)
DATA_DIR = os.path.join(BASE_DIR, '..', 'data')

def get_all_qa():
    \"\"\"Load toàn bộ Q&A từ CSV\"\"\"
    return pd.read_csv(os.path.join(DATA_DIR, 'qa_train.csv'))
from preprocess import preprocess_text, train_vectorizer
from nb_module import train_naive_bayes
from knn_module import train_knn_model  # 🆕 Import KNN

# Download NLTK data
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('punkt_tab', quiet=True)

BASE_DIR = os.path.dirname(__file__)
MODEL_DIR = os.path.join(BASE_DIR, '..', 'models')

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

    # 4. Tạo Model Naive Bayes (phân loại topic)
    nb_model = train_naive_bayes(vectorizer, df['clean_text'], df['topic'])
    
    # 5. 🆕 Tạo Model KNN (tìm câu hỏi gần nhất)
    print('\n🔍 Training KNN model...')
    knn_model = train_knn_model(
        vectorizer, 
        df['clean_text'].tolist(),  # Câu hỏi đã preprocess
        df['answer'].tolist(),       # Câu trả lời
        df['topic'].tolist(),        # Topic
        k=5                          # Số neighbors
    )

    print('\n✅ Hoàn tất! Đã tạo vectorizer.pkl, nb_model.pkl và knn_model.pkl trong thư mục models/.')

if __name__ == '__main__':
    create_pkl_files()

