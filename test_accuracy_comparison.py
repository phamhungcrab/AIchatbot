"""
🧪 Test Accuracy Comparison: Old Data vs Merged Data
Compare KNN accuracy between original training data and merged data
"""

import pandas as pd
import numpy as np
import sys
sys.path.insert(0, 'app')

from preprocess import preprocess_for_knn, train_vectorizer
from knn_module import CustomKNN

def run_knn_evaluation(train_file: str, test_file: str, name: str):
    """
    Train KNN on train_file and evaluate on test_file.
    
    Args:
        train_file: Path to training CSV
        test_file: Path to test CSV
        name: Name for this experiment
        
    Returns:
        dict with accuracy metrics
    """
    print(f"\n{'='*50}")
    print(f"🔄 Testing: {name}")
    print(f"{'='*50}")
    
    # Load data
    df_train = pd.read_csv(train_file)
    df_test = pd.read_csv(test_file)
    
    print(f"📊 Train size: {len(df_train)}, Test size: {len(df_test)}")
    
    # Preprocess questions
    train_questions = [preprocess_for_knn(q) for q in df_train['question'].tolist()]
    test_questions = [preprocess_for_knn(q) for q in df_test['question'].tolist()]
    
    train_answers = df_train['answer'].tolist()
    train_topics = df_train['topic'].tolist()
    
    test_answers = df_test['answer'].tolist()
    
    # Train vectorizer and transform
    vectorizer = train_vectorizer(train_questions)
    X_train = vectorizer.transform(train_questions)  # Shape: (n_train, n_features)
    X_test = vectorizer.transform(test_questions)    # Shape: (n_test, n_features)
    
    print(f"🔢 Feature dimension: {X_train.shape[1]}")
    
    # Train KNN
    knn = CustomKNN(k=5, metric='cosine')
    knn.fit(X_train, train_questions, train_answers, train_topics)
    
    # Evaluate
    correct = 0
    high_conf_correct = 0
    high_conf_total = 0
    
    n_test = X_test.shape[0]  # Use shape[0] for sparse matrix
    for i in range(n_test):
        pred_answer, confidence, matched_q, topic = knn.predict(X_test[i])
        
        if pred_answer == test_answers[i]:
            correct += 1
        
        # High confidence threshold (>= 0.7)
        if confidence >= 0.7:
            high_conf_total += 1
            if pred_answer == test_answers[i]:
                high_conf_correct += 1
    
    accuracy = correct / n_test * 100
    high_conf_accuracy = (high_conf_correct / high_conf_total * 100) if high_conf_total > 0 else 0
    high_conf_coverage = high_conf_total / n_test * 100
    
    print(f"\n📈 Results for {name}:")
    print(f"   Overall Accuracy: {accuracy:.2f}%")
    print(f"   High Conf (>=70%) Accuracy: {high_conf_accuracy:.2f}% (Coverage: {high_conf_coverage:.2f}%)")
    
    return {
        'name': name,
        'train_size': len(df_train),
        'accuracy': accuracy,
        'high_conf_accuracy': high_conf_accuracy,
        'high_conf_coverage': high_conf_coverage
    }

if __name__ == "__main__":
    print("🧪 KNN ACCURACY COMPARISON TEST")
    print("=" * 60)
    
    # Test with old data
    result_old = run_knn_evaluation(
        train_file='data/qa_train.csv',
        test_file='data/qa_test.csv',
        name='Original Data (1000 rows)'
    )
    
    # Test with merged data
    result_merged = run_knn_evaluation(
        train_file='data/qa_train_merged.csv',
        test_file='data/qa_test.csv',
        name='Merged Data (1239 rows)'
    )
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 SUMMARY COMPARISON")
    print("=" * 60)
    print(f"{'Metric':<30} {'Original':<15} {'Merged':<15} {'Diff':<10}")
    print("-" * 70)
    
    acc_diff = result_merged['accuracy'] - result_old['accuracy']
    sign = '+' if acc_diff > 0 else ''
    print(f"{'Overall Accuracy':<30} {result_old['accuracy']:.2f}%{'':<10} {result_merged['accuracy']:.2f}%{'':<10} {sign}{acc_diff:.2f}%")
    
    hc_diff = result_merged['high_conf_accuracy'] - result_old['high_conf_accuracy']
    sign = '+' if hc_diff > 0 else ''
    print(f"{'High Conf Accuracy (>=70%)':<30} {result_old['high_conf_accuracy']:.2f}%{'':<10} {result_merged['high_conf_accuracy']:.2f}%{'':<10} {sign}{hc_diff:.2f}%")
    
    cov_diff = result_merged['high_conf_coverage'] - result_old['high_conf_coverage']
    sign = '+' if cov_diff > 0 else ''
    print(f"{'High Conf Coverage':<30} {result_old['high_conf_coverage']:.2f}%{'':<10} {result_merged['high_conf_coverage']:.2f}%{'':<10} {sign}{cov_diff:.2f}%")
    
    print("\n✅ Test completed!")
