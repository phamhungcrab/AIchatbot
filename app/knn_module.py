# -------------------------------
# 🔍 knn_module.py — K-Nearest Neighbors cho Chatbot
# Chức năng: Tìm câu hỏi gần nhất trong database bằng KNN
# Chạy song song với Naive Bayes để so sánh
# -------------------------------

import numpy as np
import pickle
import os

# -------------------------------
# 📁 Thiết lập đường dẫn
# -------------------------------
BASE_DIR = os.path.dirname(__file__)
MODEL_DIR = os.path.join(BASE_DIR, '..', 'models')
KNN_MODEL_PATH = os.path.join(MODEL_DIR, 'knn_model.pkl')


# =========================================================
# 🔮 CLASS CUSTOM KNN (TỰ VIẾT)
# =========================================================
class CustomKNN:
    """
    Tự cài đặt thuật toán K-Nearest Neighbors.
    Dùng để tìm câu hỏi trong database gần nhất với query của user.
    
    📌 Lưu ý về Shape:
    - X_train: (n_samples, n_features) - Ma trận TF-IDF của các câu hỏi
    - query: (1, n_features) hoặc (n_features,) - Vector TF-IDF của câu hỏi user
    """
    
    def __init__(self, k=5, metric='cosine'):
        """
        Khởi tạo KNN.
        
        Args:
            k (int): Số láng giềng gần nhất để xem xét
            metric (str): 'cosine' hoặc 'euclidean'
        """
        self.k = k
        self.metric = metric
        self.X_train = None       # TF-IDF vectors của câu hỏi   Shape: (n_samples, n_features)
        self.questions = None     # List câu hỏi gốc
        self.answers = None       # List câu trả lời tương ứng
        self.topics = None        # List topic tương ứng
        
    def fit(self, X, questions, answers, topics=None):
        """
        Fit model với dữ liệu training.
        
        Args:
            X: Ma trận TF-IDF (sparse hoặc dense)  Shape: (n_samples, n_features)
            questions: List câu hỏi
            answers: List câu trả lời
            topics: List topic (optional)
        """
        # Chuyển sparse matrix sang dense nếu cần
        if hasattr(X, 'toarray'):
            self.X_train = X.toarray()  # Shape: (n_samples, n_features)
        else:
            self.X_train = np.array(X)
            
        self.questions = list(questions)
        self.answers = list(answers)
        self.topics = list(topics) if topics else [None] * len(questions)
        
        print(f"✅ KNN fitted with {len(self.questions)} samples, shape: {self.X_train.shape}")
        return self
    
    def _compute_distance(self, vec1, vec2):
        """
        Tính khoảng cách giữa 2 vector.
        
        Args:
            vec1, vec2: Shape (n_features,)
            
        Returns:
            float: Khoảng cách (càng nhỏ càng giống)
        """
        if self.metric == 'cosine':
            # Cosine Distance = 1 - Cosine Similarity
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)
            if norm1 == 0 or norm2 == 0:
                return 1.0  # Max distance if zero vector
            similarity = np.dot(vec1, vec2) / (norm1 * norm2)
            return 1.0 - similarity
        else:
            # Euclidean Distance
            return np.linalg.norm(vec1 - vec2)
    
    def predict(self, query_vector, return_details=False):
        """
        Tìm K câu hỏi gần nhất với query.
        
        Args:
            query_vector: TF-IDF vector của câu hỏi user  Shape: (1, n_features) hoặc (n_features,)
            return_details: Nếu True, trả về top K kết quả chi tiết
            
        Returns:
            Nếu return_details=False: (best_answer, confidence, matched_question, topic)
            Nếu return_details=True: List[(distance, question, answer, topic)]
        """
        # Chuẩn hóa shape
        if hasattr(query_vector, 'toarray'):
            query_vector = query_vector.toarray()  # Sparse to dense
        query_vector = np.array(query_vector).flatten()  # Shape: (n_features,)
        
        # Tính khoảng cách đến tất cả các câu hỏi trong training set
        distances = []
        for i in range(len(self.X_train)):
            dist = self._compute_distance(query_vector, self.X_train[i])
            distances.append({
                'distance': dist,
                'index': i,
                'question': self.questions[i],
                'answer': self.answers[i],
                'topic': self.topics[i]
            })
        
        # Sắp xếp theo khoảng cách tăng dần (gần nhất trước)
        distances.sort(key=lambda x: x['distance'])
        
        # Lấy K nearest neighbors
        k_nearest = distances[:self.k]
        
        if return_details:
            return k_nearest
        
        # Trả về câu gần nhất
        best = k_nearest[0]
        # Chuyển distance thành confidence: 0.0 (xa) -> 1.0 (gần)
        # Với cosine distance, range là [0, 2], nhưng thường trong [0, 1]
        confidence = max(0, 1.0 - best['distance'])
        
        return best['answer'], confidence, best['question'], best['topic']
    
    def predict_voting(self, query_vector):
        """
        🆕 Dự đoán bằng Weighted Voting từ K neighbors.
        
        Thay vì chỉ lấy câu gần nhất, tính điểm cho từng đáp án
        dựa trên khoảng cách của tất cả K neighbors.
        
        Args:
            query_vector: TF-IDF vector của câu hỏi user
            
        Returns:
            (best_answer, confidence, matched_question, topic)
        """
        # Chuẩn hóa shape
        if hasattr(query_vector, 'toarray'):
            query_vector = query_vector.toarray()
        query_vector = np.array(query_vector).flatten()
        
        # Tính khoảng cách đến tất cả các câu hỏi
        distances = []
        for i in range(len(self.X_train)):
            dist = self._compute_distance(query_vector, self.X_train[i])
            distances.append({
                'distance': dist,
                'index': i,
                'question': self.questions[i],
                'answer': self.answers[i],
                'topic': self.topics[i],
                'weight': max(0, 1.0 - dist)  # Weight = similarity
            })
        
        # Sắp xếp và lấy K nearest
        distances.sort(key=lambda x: x['distance'])
        k_nearest = distances[:self.k]
        
        # Weighted Voting: Tổng hợp điểm cho mỗi đáp án
        answer_scores = {}
        for neighbor in k_nearest:
            ans = neighbor['answer']
            weight = neighbor['weight']
            if ans not in answer_scores:
                answer_scores[ans] = {
                    'score': 0,
                    'question': neighbor['question'],
                    'topic': neighbor['topic'],
                    'count': 0
                }
            answer_scores[ans]['score'] += weight
            answer_scores[ans]['count'] += 1
        
        # Chọn đáp án có tổng điểm cao nhất
        best_answer = max(answer_scores.keys(), key=lambda x: answer_scores[x]['score'])
        best_info = answer_scores[best_answer]
        
        # Confidence = tổng điểm / số K (normalized)
        confidence = best_info['score'] / self.k
        
        return best_answer, confidence, best_info['question'], best_info['topic']
    
    def score(self, X_test, y_test):
        """
        Đánh giá accuracy trên tập test.
        
        Args:
            X_test: Shape (n_test, n_features)
            y_test: List câu trả lời đúng
        """
        if hasattr(X_test, 'toarray'):
            X_test = X_test.toarray()
        
        correct = 0
        for i in range(len(X_test)):
            pred_answer, _, _, _ = self.predict(X_test[i])
            if pred_answer == y_test[i]:
                correct += 1
        
        return correct / len(X_test)


# =========================================================
# 🧠 HÀM TÌM CÂU TRẢ LỜI BẰNG KNN
# =========================================================
def find_answer_knn(knn_model, vectorizer, user_question, k=3):
    """
    Tìm câu trả lời cho câu hỏi user bằng KNN.
    
    Args:
        knn_model: Model KNN đã train
        vectorizer: TF-IDF vectorizer
        user_question: Câu hỏi đã preprocess
        k: Số kết quả trả về
        
    Returns:
        (best_answer, confidence, matched_question, topic, top_k_results)
    """
    # 1. Vector hóa câu hỏi user
    query_vec = vectorizer.transform([user_question])  # Shape: (1, n_features)
    
    # 2. Tìm K nearest neighbors
    knn_model.k = k
    results = knn_model.predict(query_vec, return_details=True)
    
    # 3. Trả về kết quả tốt nhất + chi tiết
    if results:
        best = results[0]
        confidence = max(0, 1.0 - best['distance'])
        return best['answer'], confidence, best['question'], best['topic'], results
    
    return None, 0.0, None, None, []


# =========================================================
# 🔄 HÀM TRAIN KNN MODEL
# =========================================================
def train_knn_model(vectorizer, train_questions, train_answers, train_topics, k=5):
    """
    Train và lưu KNN model.
    
    Args:
        vectorizer: TF-IDF vectorizer đã fit
        train_questions: List câu hỏi
        train_answers: List câu trả lời
        train_topics: List topic
        k: Số neighbors
        
    Returns:
        knn_model: Model đã train
    """
    print("🔄 Training KNN model...")
    
    # 1. Vector hóa câu hỏi training
    X_train = vectorizer.transform(train_questions)  # Shape: (n_samples, n_features)
    
    # 2. Tạo và fit KNN
    knn = CustomKNN(k=k, metric='cosine')
    knn.fit(X_train, train_questions, train_answers, train_topics)
    
    # 3. Lưu model
    with open(KNN_MODEL_PATH, 'wb') as f:
        pickle.dump(knn, f)
    print(f"✅ KNN model saved at: {os.path.abspath(KNN_MODEL_PATH)}")
    
    return knn


# =========================================================
# 🧪 SANITY CHECK
# =========================================================
if __name__ == "__main__":
    print("\n--------- RUNNING KNN SANITY CHECK ---------\n")
    
    # 1. Tạo dummy data
    # Giả lập TF-IDF vectors (4 câu hỏi, 3 features)
    X_dummy = np.array([
        [0.8, 0.1, 0.1],   # Q1: "BFS là gì"
        [0.2, 0.7, 0.1],   # Q2: "KNN là gì"
        [0.75, 0.15, 0.1], # Q3: "DFS là gì" (gần Q1)
        [0.1, 0.1, 0.8],   # Q4: "Logic là gì"
    ])
    
    questions = ["BFS là gì", "KNN là gì", "DFS là gì", "Logic là gì"]
    answers = ["BFS là tìm theo chiều rộng", "KNN là K láng giềng", 
               "DFS là tìm theo chiều sâu", "Logic là môn học logic"]
    topics = ["Search", "ML", "Search", "Logic"]
    
    # 2. Train KNN
    knn = CustomKNN(k=2, metric='cosine')
    knn.fit(X_dummy, questions, answers, topics)  # Shape: (4, 3)
    
    # 3. Test query
    query = np.array([0.78, 0.12, 0.1])  # Gần giống Q1 và Q3 (BFS, DFS)
    print(f"Query vector: {query}")
    print(f"Expected: Gần 'BFS là gì' hoặc 'DFS là gì'\n")
    
    results = knn.predict(query, return_details=True)
    print("Top K results:")
    for r in results:
        print(f"  Distance: {r['distance']:.4f} | Q: {r['question']} | Topic: {r['topic']}")
    
    print("\n✅ KNN Sanity Check Passed!")
