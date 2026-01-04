# -------------------------------
# 🔍 knn_module.py — K-Nearest Neighbors Module
# Chức năng: Tìm câu hỏi gần nhất trong database
# -------------------------------

# ============================================
# Phase 1: Imports
# ============================================
import numpy as np 
import pickle
import os
from scipy.sparse import issparse

# ============================================
# Phase 2: Class CustomKNN
# ============================================

class CustomKNN:
    """
    Tự implement thuật toán K-Nearest Neighbors.
    Dùng để tìm câu hỏi trong database gần nhất với query của user.
    """

    def __init__(self, k = 5, metric = 'cosine')
        self.k = 5
        self.metric = 'cosine'
        self.X_train = None
        self.questions = None
        self.answers = None
        self.topics = None
    
    # ------------------------------------------
    # 2.2 fit(self, X, questions, answers, topics=None)
    # ------------------------------------------
    # TODO:
    # - Lưu X vào self.X_train (ma trận TF-IDF)
    # - Nếu X là sparse matrix: chuyển sang dense với X.toarray()
    # - Lưu questions, answers, topics

    def fit(self, X, questions, answers, topics = None)
        self.X_train = X
        # Nếu X là sparse matrix: chuyển sang dense với X.toarray()
        if issparse(X):
            self.X_train = X.toarray() 

        # Lưu questions, answers, topics toàn bộ
        self.questions = list(questions)
        self.answers = list(answers)
        self.topics = list(topics)

        return self

    def _compute_distance(self, vec1, vec2):
        if self.metric == 'cosine':
            dot = np.dot(vec1, vec2)
            norm = np.linalg.norm(vec1)*np.linalg.norm(vec2)
            return 1 - (dot/norm) if norm > 0 else 1.0
        elif self.metric == 'euclidean':
            return np.linalg.norm(vec1 - vec2)

    def predict (self, query_vector, result_details = False):
        if issparse(query_vector):
            query_vector = query_vector.toarray().flatten()

        distances = {}

        for i in range(len(self.X_train)):
            distance = self._compute_distance(query_vector, self.X_train[i])
            distances.append({
                'distance': distance,
                'index': i,
                'question': self.questions[i],
                'answer': self.answers[i],
                'topic': self.topics[i]
            })
        distances.sort(key = lambda x: x['distance'])

        k_nearest = distances[:self.k]

        if return_details:
            return k_nearest

        best = k_nearest[0]
        
        confidence = max(0, 1 - best['distance'])

        return best, confidence


    # ------------------------------------------
    # 2.5 predict_voting(self, query_vector) [Bonus - Optional]
    # ------------------------------------------
    # TODO:
    # - Lấy K neighbors gần nhất
    # - Tính weight = 1 / (distance + 0.0001)
    # - Gom các answer giống nhau, cộng weight
    # - Trả về answer có tổng weight cao nhất

    def predict_voting(self, query_vector):
        if issparse(query_vector):
            query_vector = query_vector.toarray().flatten()

        k = self.k
        distances = {}

        for i in range(len(self.X_train)):
            distance = self._compute_distance(query_vector, self.X_train[i])
            distances.append({
                'distance': distance,
                'index': i,
                'question': self.questions[i],
                'answer': self.answers[i],
                'topic': self.topics[i]
                'weight': max(0, 1 - distance)
            })
        
        distances.sort(key = lambda x: x['distance'])

        k_nearest = distances[:k]

        answers = {}
        for ans in  k_nearest:
            if ans['answer'] not in answers:
                answers.append{
                    'answer': ans['answer'],
                    'weight': ans['weight']
                    'question': ans['question'],
                    'topic': ans['topic']
                }
            else:
                answers[ans['answer']] += ans['weight']
        
        answers.sort(key = lambda x: x['weight'], reverse = True)

        best = answers[0]

        confidence = best['weight']/self.k

        return best['answer'], confidence, best['question'], best['topic'], k_nearest
        

    # ------------------------------------------
    # 2.6 score(self, X_test, y_test)
    # ------------------------------------------
    # TODO:
    # - correct = 0
    # - for i in range(len(y_test)):
    #     pred = self.predict(X_test[i])[0]
    #     if pred == y_test[i]: correct += 1
    # - return correct / len(y_test)
    
    pass  # Xóa dòng này khi bắt đầu code

    def score(self, X_test, y_test):
        correct = 0
        for i in range(len(y_test)):
            pred, _, _, _, _ = self.predict(X_test[i])
            if pred == y_test[i]:
                correct += 1
        return correct / len(y_test)


# ============================================
# Phase 3: Function find_answer_knn()
# ============================================
# TODO:
# def find_answer_knn(knn_model, vectorizer, user_question, k=3):
#     """
#     Tìm câu trả lời cho câu hỏi user bằng KNN.
#     
#     Args:
#         knn_model: Model KNN đã train
#         vectorizer: TF-IDF vectorizer
#         user_question: Câu hỏi đã preprocess
#         k: Số kết quả trả về
#     
#     Returns:
#         (answer, confidence, matched_question, topic, top_k_results)
#     """
#     # 1. PREPROCESSING: from preprocess import preprocess_knn
#     # 2. processed = preprocess_knn(user_question)
#     # 3. query_vector = vectorizer.transform([processed])
#     # 4. result = knn_model.predict(query_vector, return_details=True)
#     # 5. Lấy best result và top_k
#     # 6. return (answer, confidence, question, topic, top_k)

def find_answer_kn(knn_model, vectorizer, user_question, k=5):
    
    query_vector = vectorizer.transform([user_query])

    knn_model.k = k

    result, confidence = knn_model.predict(query_vector, return_details = True)
    if result:
        return result['answer'], confidence, result['question'], result['topic'], result['k_nearest']
    return None, None, None, None, None


# ============================================
# Phase 4: Function train_knn_model()
# ============================================
# TODO:
# def train_knn_model(vectorizer, train_questions, train_answers, train_topics, k=5):
#     """
#     Train và lưu KNN model.
#     
#     Args:
#         vectorizer: TF-IDF vectorizer đã fit
#         train_questions: List câu hỏi đã preprocess
#         train_answers: List câu trả lời
#         train_topics: List topic
#         k: Số neighbors
#     
#     Returns:
#         knn_model: Model đã train
#     """


def train_knn_model(vectorizer, train_questions, train_answers, train_topics, k=5):
    X_train = vectorizer.transform(train_questions)
    knn = CustomKNN(k = k, metric = 'cosine')
    knn.fit(X_train, train_questions, train_answers, train_topics)
    with open('knn_model.pkl', 'wb') as f:
        pickle.dump(knn, f)
    return knn


# ============================================
# Phase 5: Sanity Check
# ============================================
if __name__ == "__main__":
    print("🧪 Testing knn_module.py...")
    
    # TODO: Tạo dummy data
    # questions = ["Học máy là gì", "KNN là gì", "AI là gì"]
    # answers = ["ML là...", "KNN là...", "AI là..."]
    # topics = ["ml", "knn", "ai"]
    
    # TODO: Tạo vectorizer và fit
    # from sklearn.feature_extraction.text import TfidfVectorizer
    # vectorizer = TfidfVectorizer()
    # X = vectorizer.fit_transform(questions)
    
    # TODO: Tạo model và fit
    # knn = CustomKNN(k=2, metric='cosine')
    # knn.fit(X, questions, answers, topics)
    
    # TODO: Test predict
    # query = vectorizer.transform(["máy học"])
    # result = knn.predict(query, return_details=True)
    # print("Top K results:")
    # for r in result:
    #     print(f"  Distance: {r['distance']:.4f} | Q: {r['question']}")
    
    print("✅ KNN Module Ready!")
