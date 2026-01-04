# simple_evaluation.py — Đánh giá đơn giản cho trình bày
# Gộp từ compare_knn_methods.py và compare_nb_methods.py

import numpy as np
import pandas as pd
import pickle
import os
from sklearn.metrics.pairwise import cosine_similarity

from preprocess import preprocess_text, preprocess_for_knn
from nb_module import predict_topic
from knn_module import CustomKNN
from find_answer import find_best_answer

BASE_DIR = os.path.dirname(__file__)
MODEL_DIR = os.path.join(BASE_DIR, '..', 'models')
DATA_DIR = os.path.join(BASE_DIR, '..', 'data')


# ============================================
# 1. NB Pipeline: So sánh Cosine+Jaccard vs Pure Cosine
# ============================================

def find_answer_pure_cosine(vectorizer, question, df_topic):
    """Tìm câu trả lời chỉ bằng Pure Cosine Similarity (không Jaccard)"""
    if df_topic.empty:
        return None, 0.0, None
    
    corpus = df_topic['question'].tolist()
    all_vectors = vectorizer.transform(corpus + [question])
    
    user_vector = all_vectors[-1]
    database_vectors = all_vectors[:-1]
    
    cosine_sim = cosine_similarity(user_vector, database_vectors).flatten()
    
    best_idx = np.argmax(cosine_sim)
    best_score = cosine_sim[best_idx]
    
    return df_topic.iloc[best_idx]['answer'], best_score, df_topic.iloc[best_idx]['question']


def compare_nb_methods():
    """So sánh NB + Cosine+Jaccard vs NB + Pure Cosine"""
    print("\n" + "=" * 60)
    print("🔬 PHƯƠNG PHÁP 1: NAIVE BAYES PIPELINE")
    print("=" * 60)
    
    # Load models
    with open(os.path.join(MODEL_DIR, 'vectorizer.pkl'), 'rb') as f:
        vectorizer = pickle.load(f)
    with open(os.path.join(MODEL_DIR, 'nb_model.pkl'), 'rb') as f:
        nb_model = pickle.load(f)
    
    # Load data
    df_test = pd.read_csv(os.path.join(DATA_DIR, 'qa_test.csv'))
    df_train = pd.read_csv(os.path.join(DATA_DIR, 'qa_train.csv'))
    df_train['clean_question'] = df_train['question'].apply(preprocess_text)
    
    print(f"📊 Test: {len(df_test)}, Train: {len(df_train)}")
    
    # Evaluate
    cosine_jaccard_correct = 0
    pure_cosine_correct = 0
    
    for idx, row in df_test.iterrows():
        clean_q = preprocess_text(row['question'])
        true_answer = row['answer']
        
        # NB predict topic
        pred_topic, nb_conf = predict_topic(nb_model, vectorizer, clean_q)
        df_topic = df_train[df_train['topic'] == pred_topic]
        if df_topic.empty:
            df_topic = df_train
        
        # Method 1: Cosine + Jaccard
        ans1, _, _ = find_best_answer(vectorizer, clean_q, df_topic, 
                                       original_query=row['question'], threshold=0.0)
        if ans1 == true_answer:
            cosine_jaccard_correct += 1
        
        # Method 2: Pure Cosine
        ans2, _, _ = find_answer_pure_cosine(vectorizer, clean_q, df_topic)
        if ans2 == true_answer:
            pure_cosine_correct += 1
    
    # Results
    cj_acc = cosine_jaccard_correct / len(df_test) * 100
    pc_acc = pure_cosine_correct / len(df_test) * 100
    
    print(f"\n📌 Kết quả NB Pipeline:")
    print(f"   🔹 Cosine + Jaccard:  {cosine_jaccard_correct}/{len(df_test)} = {cj_acc:.2f}%")
    print(f"   🔹 Pure Cosine:       {pure_cosine_correct}/{len(df_test)} = {pc_acc:.2f}%")
    
    if pc_acc > cj_acc:
        print(f"   ✅ Pure Cosine tốt hơn {pc_acc - cj_acc:.2f}%")
    elif cj_acc > pc_acc:
        print(f"   ✅ Cosine+Jaccard tốt hơn {cj_acc - pc_acc:.2f}%")
    else:
        print(f"   🤝 Cả hai phương pháp như nhau!")
    
    return {'cosine_jaccard': cj_acc, 'pure_cosine': pc_acc}


# ============================================
# 2. KNN Pipeline: So sánh Top-1 vs Weighted Voting
# ============================================

def compare_knn_methods():
    """So sánh KNN Top-1 vs KNN Weighted Voting"""
    print("\n" + "=" * 60)
    print("🔬 PHƯƠNG PHÁP 2: KNN PIPELINE")
    print("=" * 60)
    
    # Load model
    with open(os.path.join(MODEL_DIR, 'vectorizer.pkl'), 'rb') as f:
        vectorizer = pickle.load(f)
    with open(os.path.join(MODEL_DIR, 'knn_model.pkl'), 'rb') as f:
        knn_model = pickle.load(f)
    
    # Load test data
    df_test = pd.read_csv(os.path.join(DATA_DIR, 'qa_test.csv'))
    print(f"📊 Test samples: {len(df_test)}")
    
    # Evaluate both methods
    top1_correct = 0
    voting_correct = 0
    
    for idx, row in df_test.iterrows():
        clean_q = preprocess_for_knn(row['question'])
        true_answer = row['answer']
        
        query_vec = vectorizer.transform([clean_q])
        
        # Method 1: Top-1 (nearest neighbor)
        knn_model.k = 5
        answer1, conf1, _, _ = knn_model.predict(query_vec)
        if answer1 == true_answer:
            top1_correct += 1
        
        # Method 2: Weighted Voting
        knn_model.k = 5
        answer2, conf2, _, _ = knn_model.predict_voting(query_vec)
        if answer2 == true_answer:
            voting_correct += 1
    
    # Results
    top1_acc = top1_correct / len(df_test) * 100
    voting_acc = voting_correct / len(df_test) * 100
    
    print(f"\n📌 Kết quả KNN Pipeline:")
    print(f"   🔹 Top-1 (Nearest):    {top1_correct}/{len(df_test)} = {top1_acc:.2f}%")
    print(f"   🔹 Weighted Voting:    {voting_correct}/{len(df_test)} = {voting_acc:.2f}%")
    
    if voting_acc > top1_acc:
        print(f"   ✅ Weighted Voting tốt hơn {voting_acc - top1_acc:.2f}%")
    elif top1_acc > voting_acc:
        print(f"   ✅ Top-1 tốt hơn {top1_acc - voting_acc:.2f}%")
    else:
        print(f"   🤝 Cả hai phương pháp như nhau!")
    
    return {'top1': top1_acc, 'voting': voting_acc}


# ============================================
# Main
# ============================================

def main():
    print("=" * 60)
    print("📊 ĐÁNH GIÁ MÔ HÌNH CHATBOT")
    print("=" * 60)
    print("\n2 Pipeline chính:")
    print("  1. NB: Naive Bayes → Phân loại topic → Jaccard/Cosine → Answer")
    print("  2. KNN: K-Nearest Neighbors → Top-1 hoặc Voting → Answer")
    
    nb_results = compare_nb_methods()
    knn_results = compare_knn_methods()
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 TỔNG KẾT")
    print("=" * 60)
    print("\n┌────────────────────────────────────────────────────────┐")
    print("│ Pipeline          │ Method 1        │ Method 2        │")
    print("├────────────────────────────────────────────────────────┤")
    print(f"│ NB Pipeline       │ Cosine+Jaccard  │ Pure Cosine     │")
    print(f"│                   │ {nb_results['cosine_jaccard']:.2f}%          │ {nb_results['pure_cosine']:.2f}%          │")
    print("├────────────────────────────────────────────────────────┤")
    print(f"│ KNN Pipeline      │ Top-1 Nearest   │ Weighted Voting │")
    print(f"│                   │ {knn_results['top1']:.2f}%          │ {knn_results['voting']:.2f}%          │")
    print("└────────────────────────────────────────────────────────┘")


if __name__ == "__main__":
    main()
