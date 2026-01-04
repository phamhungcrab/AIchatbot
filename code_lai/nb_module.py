# -------------------------------
# 🤖 nb_module.py — Naive Bayes Module
# TODO: Code lại từ đầu
# -------------------------------

# Phase 1: Imports
# TODO: import numpy, pickle, os

import numpy as np 
import pickle
import os
from scipy.sparse import issparse

MODEL_PATH = 'models/nb_model.pkl'

def predict_topic(nb_model, vectorizer, text):

    X = vectorizer.transform([text])

    predicted_topic = nb_model.predict(X)[0]

    probs = nb_model.predict_proba(X)[0]
    confidence = np.max(probs)
    return predicted_topic, round(float(confidence), 4)
class CustomMultinomialNB:
    def __init__(self):
        self.alpha = 1.0
        self.class_log_prior_ = None
        self.feature_count_ = None
        self.class_count_ = None

    def fit(self, X, y):
        y = np.array(y)

        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)

        if issparse(X): 
            X = X.toarray()
        n_features = X.shape[1]

        self.class_log_prior = np.zeros(n_classes)
        self.feature_log_prob = np.zeros((n_classes, n_features))
        self.class_count_ = np.zeros(n_classes)

        for idx, c in enumerate(self.classes_):
            X_c = X[y == c]
            
            self.class_log_prior[idx] = np.log(X_c.shape[0]/X.shape[0])
            #Tính tổng số lượng từng từ có mặt trong một class
            count_word_in_class = X_c.sum(axis = 0) + self.alpha

            total_count_in_class = count_word_in_class.sum()

            self.feature_log_prob[idx, :] = np.log(count_word_in_class/total_count_in_class)

        
        return self

    def predict_log_proba(self, X):
        if issparse(X):
            X = X.toarray()

        n_samples = X.shape[0]
        n_classes = len(self.classes_)

        log_proba = np.zeros((n_samples, n_classes))

        for sample_idx in range(n_samples):
            for class_idx in range(n_classes):

                current_log_proba = self.class_log_prior[class_idx]

                for word_idx in range(X.shape[1]):
                    word_count = X[sample_idx, word_idx]
                    current_log_proba += word_count * self.feature_log_prob[class_idx, word_idx]

                log_proba[sample_idx, class_idx] = current_log_proba
        
        return log_proba

    def predict_proba(self, X):
        jll = self.predict_log_proba(X)

        #Stable làm biến đỡ bé
        #SoftMax, chuẩn hoá
        
        jll_stable = jll - jll.max(axis = 1, keepdims = True)
        exp_jll = np.exp(jll_stable)
        prob = exp_jll/exp_jll.sum(axis = 1, keepdims = True)

        return prob

    
    def predict(self, X):
        jll = self.predict_log_proba(X)
        return self.classes_[np.argmax(jll, axis = 1)]



def train_naive_bayes(vectorizer, train_texts, train_labels):
    vectorizer.set_params(
        max_features = 800,
        ngram_range = (1,2),
        min_df = 1
    )

    X_train = vectorizer.fit_transform(train_texts)

    nb_model = CustomMultinomialNB()
    nb_model.fit(X_train, train_labels)

    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(nb_model, f)

    return nb_model, vectorizer


# Sanity Check
if __name__ == "__main__":
    print("🧪 Testing nb_module.py...")
    print("=" * 50)
    
    # Tạo dummy data
    # 5 documents, 4 features (từ trong vocab)
    X_dummy = np.array([
        [2, 1, 0, 3],  # doc0 - class A
        [1, 2, 0, 2],  # doc1 - class A
        [0, 0, 3, 1],  # doc2 - class B
        [0, 1, 2, 0],  # doc3 - class B
        [1, 0, 0, 4],  # doc4 - class A
    ])
    y_dummy = np.array(['thời_tiết', 'thời_tiết', 'du_lịch', 'du_lịch', 'thời_tiết'])
    
    print(f"📊 X_dummy shape: {X_dummy.shape}")  # (5, 4)
    print(f"📊 y_dummy shape: {y_dummy.shape}")  # (5,)
    
    # Test fit()
    print("\n🔧 Testing fit()...")
    model = CustomMultinomialNB()
    model.fit(X_dummy, y_dummy)
    
    print(f"   classes_: {model.classes_}")  # ['du_lịch', 'thời_tiết']
    print(f"   class_log_prior shape: {model.class_log_prior.shape}")  # (2,)
    print(f"   feature_log_prob shape: {model.feature_log_prob.shape}")  # (2, 4)
    
    # Test predict_log_proba()
    print("\n🔧 Testing predict_log_proba()...")
    X_test = np.array([[1, 1, 0, 2]])  # 1 document mới
    log_proba = model.predict_log_proba(X_test)
    print(f"   Input shape: {X_test.shape}")  # (1, 4)
    print(f"   Output shape: {log_proba.shape}")  # (1, 2)
    print(f"   log_proba: {log_proba}")
    
    # Test predict_proba()
    print("\n🔧 Testing predict_proba()...")
    proba = model.predict_proba(X_test)
    print(f"   proba shape: {proba.shape}")  # (1, 2)
    print(f"   proba: {proba}")
    print(f"   Sum of proba (should be 1.0): {proba.sum():.4f}")
    
    # Test predict()
    print("\n🔧 Testing predict()...")
    prediction = model.predict(X_test)
    print(f"   prediction: {prediction}")  # ['thời_tiết'] hoặc ['du_lịch']
    
    # Test với nhiều samples
    print("\n🔧 Testing with multiple samples...")
    X_multi = np.array([
        [2, 1, 0, 3],  # Giống thời_tiết
        [0, 0, 3, 1],  # Giống du_lịch
    ])
    predictions = model.predict(X_multi)
    print(f"   predictions: {predictions}")
    
    print("\n" + "=" * 50)
    print("✅ All sanity checks passed!")
