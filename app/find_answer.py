# -------------------------------
# 🔎 answer_finder.py — Thay thế cho knn_module.py
# Chức năng: Tìm câu trả lời phù hợp nhất dựa trên độ tương đồng Cosine
# -------------------------------

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def find_best_answer(vectorizer, question, df_topic, original_query=None, threshold=0.5):
    """
    Tìm câu trả lời tốt nhất bằng cách so sánh Cosine Similarity 
    giữa câu hỏi người dùng và danh sách câu hỏi trong chủ đề (df_topic).
    
    Tham số:
        vectorizer: Mô hình TF-IDF đã load (vectorizer.pkl).
        question: Câu hỏi người dùng (đã pre-process & expand) -> Dùng cho Cosine.
        df_topic: DataFrame chứa các câu hỏi thuộc chủ đề đã dự đoán.
        original_query: Câu hỏi gốc của người dùng (chưa expand) -> Dùng cho Jaccard.
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
    
    # 4. Tính độ tương đồng Cosine
    user_vector = all_vectors[-1]
    database_vectors = all_vectors[:-1]
    cosine_sim = cosine_similarity(user_vector, database_vectors).flatten()
    
    # 5. Re-ranking bằng Jaccard Similarity (từ preprocess)
    from preprocess import calculate_jaccard_similarity
    
    # Lấy Top 15 ứng viên có Cosine cao nhất để kiểm tra kỹ hơn
    top_k = 15
    # Lấy indices của top k phần tử (sắp xếp giảm dần)
    top_indices = np.argsort(cosine_sim)[-top_k:][::-1]
    
    best_score = -1.0
    best_idx = -1
    
    # Quyết định dùng text nào để tính Jaccard
    # Nếu có original_query (ngắn gọn, chưa expand) thì dùng nó sẽ chính xác hơn
    query_for_jaccard = original_query if original_query else question
    
    # 🔥 Phát hiện câu hỏi so sánh
    COMPARISON_KEYWORDS = {'khác', 'khác gì', 'khác nhau', 'so sánh', 'so với', 'khác biệt', 'điểm khác'}
    query_lower = query_for_jaccard.lower()
    is_comparison_query = any(kw in query_lower for kw in COMPARISON_KEYWORDS)
    
    for idx in top_indices:
        cosine_score = cosine_sim[idx]
        
        # Nếu cosine quá thấp thì bỏ qua luôn
        if cosine_score < 0.1:
            continue
            
        candidate_question = corpus[idx]
        candidate_lower = candidate_question.lower()
        
        # Tính Jaccard (so khớp từ khóa bất chấp thứ tự)
        jaccard_score = calculate_jaccard_similarity(query_for_jaccard, candidate_question)
        
        # 🔥 Công thức: 70% Cosine + 30% Jaccard
        final_score = 0.7 * cosine_score + 0.3 * jaccard_score
        
        # 🎯 BOOST cho câu hỏi SO SÁNH
        # Nếu user hỏi so sánh VÀ candidate cũng chứa từ khóa so sánh → boost cao
        if is_comparison_query:
            candidate_is_comparison = any(kw in candidate_lower for kw in COMPARISON_KEYWORDS)
            if candidate_is_comparison:
                # Candidate phù hợp loại câu hỏi → boost lên 0.15
                final_score += 0.15
            else:
                # Candidate là câu hỏi định nghĩa ("X là gì") → giảm điểm
                if "là gì" in candidate_lower or candidate_lower.endswith("là gì?"):
                    final_score -= 0.10
        
        # 🎯 BOOST: Nếu Cosine rất cao (>= 0.7) → boost thêm
        if cosine_score >= 0.7:
            final_score = max(final_score, 0.90)  # Đảm bảo ít nhất 90%
        elif cosine_score >= 0.5:
            final_score = max(final_score, 0.80)  # Đảm bảo ít nhất 80%
        
        # Giới hạn max = 1.0
        final_score = min(final_score, 1.0)
        
        if final_score > best_score:
            best_score = final_score
            best_idx = idx
            
    # 6. Kiểm tra ngưỡng tin cậy (Threshold)
    if best_score < threshold:
        return None, best_score, None
    
    # 7. Lấy kết quả
    best_answer = df_topic.iloc[best_idx]['answer']
    matched_question = df_topic.iloc[best_idx]['question']
    
    return best_answer, best_score, matched_question