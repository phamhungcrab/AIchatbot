# -------------------------------
# 🔎 answer_finder.py — Thay thế cho knn_module.py
# Chức năng: Tìm câu trả lời phù hợp nhất dựa trên độ tương đồng Cosine
# -------------------------------

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def find_best_answer(vectorizer, question, df_topic, threshold=0.5):
    """
    Tìm câu trả lời tốt nhất bằng cách so sánh Cosine Similarity 
    giữa câu hỏi người dùng và danh sách câu hỏi trong chủ đề (df_topic).
    
    Tham số:
        vectorizer: Mô hình TF-IDF đã load (vectorizer.pkl).
        question: Câu hỏi người dùng (đã pre-process).
        df_topic: DataFrame chứa các câu hỏi thuộc chủ đề đã dự đoán.
        threshold: Ngưỡng độ tin cậy tối thiểu.
        
    Trả về:
        (answer, similarity_score, matched_question)
    """
    
    # 1. Nếu không có dữ liệu trong chủ đề này, trả về None ngay
    if df_topic.empty:
        return None, 0.0, None
    
    # 2. Lấy danh sách câu hỏi mẫu từ DB
    corpus = df_topic['question'].tolist()
    
    # 3. Gộp câu hỏi người dùng vào cuối danh sách để vector hóa chung
    # (Cách này đảm bảo tính toán đúng trên cùng không gian vector)
    all_vectors = vectorizer.transform(corpus + [question])
    
    # 4. Tính độ tương đồng giữa câu hỏi user (vector cuối) với các câu mẫu (các vector trước đó)
    user_vector = all_vectors[-1]
    database_vectors = all_vectors[:-1]
    
    cosine_sim = cosine_similarity(user_vector, database_vectors).flatten()
    
    # 5. Tìm vị trí có độ tương đồng cao nhất
    max_sim = float(np.max(cosine_sim))
    best_idx = int(np.argmax(cosine_sim))
    
    # 6. Kiểm tra ngưỡng tin cậy (Threshold)
    if max_sim < threshold:
        return None, max_sim, None
    
    # 7. Lấy kết quả
    best_answer = df_topic.iloc[best_idx]['answer']
    matched_question = df_topic.iloc[best_idx]['question']
    
    return best_answer, max_sim, matched_question