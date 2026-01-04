# -------------------------------
# 📊 evaluate_models.py — Đánh giá Model NB + KNN với Thang đo Confidence Hiện đại
# Mục đích: Test trên bộ dữ liệu validation và tạo báo cáo tổng hợp kết quả
# -------------------------------

import numpy as np
import pandas as pd
import pickle
import os
import json
from datetime import datetime

# Import các module nội bộ
from preprocess import preprocess_text, preprocess_for_knn, train_vectorizer
from nb_module import CustomMultinomialNB, predict_topic
from knn_module import CustomKNN, find_answer_knn
from find_answer import find_best_answer
from confidence_utils import UnifiedCalibrator, NaiveBayesCalibrator, KNNCalibrator

# =========================================================
# 📁 THIẾT LẬP ĐƯỜNG DẪN
# =========================================================
BASE_DIR = os.path.dirname(__file__)
MODEL_DIR = os.path.join(BASE_DIR, '..', 'models')
DATA_DIR = os.path.join(BASE_DIR, '..', 'data')
RESULTS_DIR = os.path.join(BASE_DIR, '..', 'results')

# Tạo thư mục results nếu chưa tồn tại
os.makedirs(RESULTS_DIR, exist_ok=True)

# =========================================================
# 🔧 1. LOAD MODELS VÀ DỮ LIỆU
# =========================================================
def load_models():
    """Load các model đã train từ file .pkl"""
    vectorizer_path = os.path.join(MODEL_DIR, 'vectorizer.pkl')
    nb_model_path = os.path.join(MODEL_DIR, 'nb_model.pkl')
    knn_model_path = os.path.join(MODEL_DIR, 'knn_model.pkl')
    
    with open(vectorizer_path, 'rb') as f:
        vectorizer = pickle.load(f)
    with open(nb_model_path, 'rb') as f:
        nb_model = pickle.load(f)
    with open(knn_model_path, 'rb') as f:
        knn_model = pickle.load(f)
    
    print("✅ Đã load: vectorizer, nb_model, knn_model")
    return vectorizer, nb_model, knn_model


def load_validation_data():
    """Load dữ liệu test từ CSV"""
    valid_path = os.path.join(DATA_DIR, 'qa_test.csv')
    df = pd.read_csv(valid_path)
    
    # Preprocess các câu hỏi
    df['clean_question'] = df['question'].apply(preprocess_text)
    
    print(f"📊 Đã load {len(df)} mẫu validation")
    return df


# =========================================================
# 📈 2. THANG ĐO CONFIDENCE HIỆN ĐẠI
# =========================================================
def calculate_confidence_metrics(predictions, ground_truth, confidences):
    """
    Tính toán các thang đo confidence hiện đại:
    - Accuracy
    - Average Confidence (Mean)
    - Calibration Error (ECE - Expected Calibration Error)
    - Confidence vs Accuracy Correlation
    - Coverage at Threshold
    """
    # 1. Basic Metrics
    predictions = np.array(predictions)
    ground_truth = np.array(ground_truth)
    confidences = np.array(confidences)
    
    correct = predictions == ground_truth
    accuracy = np.mean(correct)
    avg_confidence = np.mean(confidences)
    
    # 2. ECE (Expected Calibration Error) - Đo độ tin cậy của confidence
    # Chia confidence thành 10 bins, tính |accuracy - confidence| trung bình
    n_bins = 10
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    
    for i in range(n_bins):
        in_bin = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i + 1])
        if np.sum(in_bin) > 0:
            bin_accuracy = np.mean(correct[in_bin])
            bin_confidence = np.mean(confidences[in_bin])
            bin_weight = np.sum(in_bin) / len(confidences)
            ece += bin_weight * abs(bin_accuracy - bin_confidence)
    
    # 3. Coverage at different thresholds
    thresholds = [0.3, 0.5, 0.7, 0.9]
    coverage = {}
    accuracy_at_threshold = {}
    
    for thresh in thresholds:
        mask = confidences >= thresh
        coverage[thresh] = np.mean(mask) * 100  # % samples above threshold
        if np.sum(mask) > 0:
            accuracy_at_threshold[thresh] = np.mean(correct[mask]) * 100
        else:
            accuracy_at_threshold[thresh] = 0.0
    
    return {
        'accuracy': accuracy * 100,                          # % đúng
        'average_confidence': avg_confidence * 100,          # % trung bình confidence
        'expected_calibration_error': ece * 100,             # ECE (%) - càng thấp càng tốt
        'coverage_at_threshold': coverage,                   # % số mẫu >= threshold
        'accuracy_at_threshold': accuracy_at_threshold,      # Accuracy với các mẫu >= threshold
        'total_samples': len(predictions)
    }


# =========================================================
# 🧪 3. ĐÁNH GIÁ NAIVE BAYES (TOPIC CLASSIFICATION)
# =========================================================
def evaluate_naive_bayes(nb_model, vectorizer, df, df_train, calibrator=None):
    """
    Đánh giá Naive Bayes trên task phân loại topic VÀ trả lời đúng.
    
    📌 NB Pipeline:
    1. NB phân loại topic
    2. find_best_answer tìm đáp án trong topic đó
    
    📌 Metrics mới:
    - topic_accuracy: Phân loại đúng topic
    - answer_accuracy: Trả đúng đáp án (exact match)
    """
    print("\n" + "="*60)
    print("🔬 ĐÁNH GIÁ NAIVE BAYES (TOPIC + ANSWER)")
    print("="*60)
    
    topic_predictions = []
    answer_predictions = []
    answer_matches = []
    answer_confidences = []  # 🆕 Confidence của câu trả lời (từ similarity score)
    details = []
    
    for idx, row in df.iterrows():
        clean_q = row['clean_question']
        true_topic = row['topic']
        true_answer = row['answer']
        
        # 1. NB predict topic
        pred_topic, topic_conf = predict_topic(nb_model, vectorizer, clean_q)
        
        # 2. find_best_answer trong topic đó
        df_topic = df_train[df_train['topic'] == pred_topic]
        if df_topic.empty:
            df_topic = df_train  # Fallback to all data
        
        pred_answer, sim_score, matched_q = find_best_answer(
            vectorizer, clean_q, df_topic,
            original_query=row['question'], threshold=0.0  # Không filter, lấy hết
        )
        
        # 3. Check exact match
        answer_match = (pred_answer == true_answer) if pred_answer else False
        
        # 4. 🆕 Confidence = topic_conf × similarity_score (kết hợp cả 2)
        raw_sim = sim_score if sim_score else 0.0
        # Combined score: topic_conf * similarity
        combined_score = topic_conf * raw_sim
        # Sigmoid scaling: chuyển từ [0-1] về [0-1] với midpoint hợp lý
        k = 10.0
        midpoint = 0.4  # Midpoint cho combined score
        answer_conf = 1.0 / (1.0 + np.exp(-k * (combined_score - midpoint)))
        
        topic_predictions.append(pred_topic)
        answer_predictions.append(pred_answer)
        answer_matches.append(answer_match)
        answer_confidences.append(answer_conf)
        details.append({
            'question': row['question'][:50] + '...' if len(row['question']) > 50 else row['question'],
            'true_topic': true_topic,
            'predicted_topic': pred_topic,
            'topic_correct': pred_topic == true_topic,
            'answer_correct': answer_match,
            'topic_confidence': topic_conf,
            'answer_confidence': answer_conf  # 🆕 Độ tin cậy câu trả lời
        })
    
    # 🆕 Tính metrics dựa trên ANSWER confidence (không phải topic confidence)
    answer_matches_arr = np.array(answer_matches)
    answer_confidences_arr = np.array(answer_confidences)
    
    # Accuracy at different thresholds
    thresholds = [0.3, 0.5, 0.7, 0.9]
    coverage = {}
    accuracy_at_threshold = {}
    
    for thresh in thresholds:
        mask = answer_confidences_arr >= thresh
        coverage[thresh] = np.mean(mask) * 100
        if np.sum(mask) > 0:
            accuracy_at_threshold[thresh] = np.mean(answer_matches_arr[mask]) * 100
        else:
            accuracy_at_threshold[thresh] = 0.0
    
    # Overall metrics
    topic_accuracy = np.mean(np.array(topic_predictions) == df['topic'].values) * 100
    answer_accuracy = np.mean(answer_matches) * 100
    avg_answer_confidence = np.mean(answer_confidences) * 100
    
    print(f"   Topic Accuracy: {topic_accuracy:.2f}%")
    print(f"   Answer Accuracy: {answer_accuracy:.2f}%")
    print(f"   Avg Answer Confidence: {avg_answer_confidence:.2f}%")
    
    # Build metrics dict
    answer_metrics = {
        'topic_accuracy': topic_accuracy,
        'answer_accuracy': answer_accuracy,
        'average_answer_confidence': avg_answer_confidence,
        'coverage_at_threshold': coverage,
        'accuracy_at_threshold': accuracy_at_threshold,
        'total_samples': len(df)
    }
    
    return {
        'model': 'Naive Bayes Pipeline (Topic + Answer)',
        'technique': {
            'algorithm': 'NB Topic Classification + Cosine+Jaccard Answer Retrieval',
            'nb_formula': 'P(topic|X) ∝ P(topic) × ∏ P(word_i|topic)',
            'answer_formula': 'Score = 0.7×Cosine + 0.3×Jaccard',
            'smoothing': 'Laplace Smoothing (alpha=0.1)',
            'vectorizer': 'TF-IDF (800 features, unigram+bigram, sublinear_tf=True)',
        },
        'confidence_type': 'Answer Confidence (từ similarity score của find_answer)',
        'metrics': answer_metrics,
        'sample_results': details[:10]
    }


# =========================================================
# 🔍 4. ĐÁNH GIÁ KNN (ANSWER RETRIEVAL)
# =========================================================
def evaluate_knn(knn_model, vectorizer, df, calibrator=None):
    """
    Đánh giá KNN trên task tìm câu trả lời.
    
    📌 KNN sử dụng:
    - Cosine Distance = 1 - Cosine Similarity
    - Raw Confidence = 1 - Distance = Cosine Similarity
    - K = 5 neighbors (mặc định)
    
    📌 Confidence Calibration (Sigmoid Scaling):
    - raw_conf = cosine_similarity (thường 0.2-0.6 với TF-IDF)
    - calibrated = sigmoid(k * (raw_conf - midpoint))
    - Chuyển similarity về scale [0,1] hợp lý hơn
    """
    print("\n" + "="*60)
    print("🔍 ĐÁNH GIÁ KNN (ANSWER RETRIEVAL)")
    print("="*60)
    
    predictions = []
    raw_confidences = []
    calibrated_confidences = []
    exact_matches = []
    details = []
    
    for idx, row in df.iterrows():
        # Dùng preprocess_for_knn thay vì preprocess_text
        clean_q = preprocess_for_knn(row['question'])
        true_answer = row['answer']
        
        # Tìm câu trả lời bằng KNN
        pred_answer, raw_conf, matched_q, topic, top_k = find_answer_knn(
            knn_model, vectorizer, clean_q, k=3
        )
        
        # Calibrate confidence
        if calibrator:
            calibrated_conf = calibrator.calibrate_knn(raw_conf)
        else:
            calibrated_conf = raw_conf
        
        # So sánh exact match
        is_exact = (pred_answer == true_answer)
        
        predictions.append(pred_answer)
        raw_confidences.append(raw_conf)
        calibrated_confidences.append(calibrated_conf)
        exact_matches.append(is_exact)
        details.append({
            'question': row['question'][:50] + '...' if len(row['question']) > 50 else row['question'],
            'true_answer': true_answer[:50] + '...' if len(true_answer) > 50 else true_answer,
            'predicted_answer': pred_answer[:50] + '...' if pred_answer and len(pred_answer) > 50 else pred_answer,
            'matched_question': matched_q[:50] + '...' if matched_q and len(matched_q) > 50 else matched_q,
            'raw_confidence': raw_conf,
            'calibrated_confidence': calibrated_conf,
            'exact_match': is_exact,
            'cosine_distance': round(1 - raw_conf, 4)
        })
    
    # Tính metrics cho CẢ raw và calibrated
    exact_matches_arr = np.array(exact_matches)
    raw_confidences_arr = np.array(raw_confidences)
    calibrated_confidences_arr = np.array(calibrated_confidences)
    
    def compute_knn_metrics(confidences, exact_matches):
        thresholds = [0.3, 0.5, 0.7, 0.9]
        coverage = {}
        accuracy_at_threshold = {}
        
        for thresh in thresholds:
            mask = confidences >= thresh
            coverage[thresh] = np.mean(mask) * 100
            if np.sum(mask) > 0:
                accuracy_at_threshold[thresh] = np.mean(exact_matches[mask]) * 100
            else:
                accuracy_at_threshold[thresh] = 0.0
        
        return {
            'exact_match_accuracy': np.mean(exact_matches) * 100,
            'average_confidence': np.mean(confidences) * 100,
            'coverage_at_threshold': coverage,
            'accuracy_at_threshold': accuracy_at_threshold,
            'total_samples': len(confidences)
        }
    
    raw_metrics = compute_knn_metrics(raw_confidences_arr, exact_matches_arr)
    calibrated_metrics = compute_knn_metrics(calibrated_confidences_arr, exact_matches_arr)
    
    return {
        'model': 'KNN (Answer Retrieval)',
        'technique': {
            'algorithm': 'Custom K-Nearest Neighbors',
            'distance_metric': 'Cosine Distance = 1 - (A·B)/(||A||×||B||)',
            'raw_confidence_formula': 'Raw Confidence = 1 - Cosine Distance = Cosine Similarity',
            'k_neighbors': 5,
            'vectorizer': 'TF-IDF (800 features, unigram+bigram, sublinear_tf=True)',
            'preprocessing': [
                'Lowercase',
                'Xóa ký tự đặc biệt (KHÔNG xóa số)',
                'PyVi Tokenizer (tách từ tiếng Việt)',
                'LIGHT_STOPWORDS (giữ từ khóa quan trọng)',
                'Synonym Expansion (mở rộng với từ đồng nghĩa)'
            ]
        },
        'confidence_calibration': {
            'method': 'Sigmoid Scaling',
            'formula': 'calibrated = 1 / (1 + exp(-k × (similarity - midpoint)))',
            'k': calibrator.knn_calibrator.k if calibrator else 10.0,
            'midpoint': calibrator.knn_calibrator.midpoint if calibrator else 0.4,
            'interpretation': 'Chuyển similarity từ [0.2-0.6] về [0-1] hợp lý hơn'
        },
        'metrics': {
            'raw': raw_metrics,
            'calibrated': calibrated_metrics
        },
        'sample_results': details[:10]
    }


# =========================================================
# 📝 5. TẠO BÁO CÁO TỔNG HỢP
# =========================================================
def generate_report(nb_results, knn_results):
    """Tạo báo cáo tổng hợp dạng dictionary và lưu ra file"""
    
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    report = {
        'evaluation_timestamp': timestamp,
        'summary': {
            'description': 'Đánh giá hệ thống AI Chatbot với 2 model: Naive Bayes (phân loại topic) và KNN (tìm câu trả lời)',
            'data_source': 'qa_valid.csv',
            'total_test_samples': nb_results['metrics']['total_samples'],
        },
        'preprocessing_pipeline': {
            'description': 'Quy trình tiền xử lý văn bản tiếng Việt',
            'steps': [
                '1. Lowercase: Chuyển về chữ thường',
                '2. Special Char Removal: Xóa ký tự đặc biệt bằng Regex r"[^\\w\\s]"',
                '3. Number Removal: Xóa số bằng Regex r"\\d+"',
                '4. PyVi Tokenizer: Tách từ tiếng Việt (ViTokenizer.tokenize)',
                '5. Stopword Removal: Lọc các từ dừng tiếng Việt (52 từ)',
            ],
            'vectorization': {
                'method': 'TF-IDF (Term Frequency - Inverse Document Frequency)',
                'params': {
                    'max_features': 800,
                    'ngram_range': '(1, 2) - unigram + bigram',
                    'min_df': 1,
                    'sublinear_tf': True,
                    'formula': 'TF-IDF(t,d) = (1 + log(tf)) × log(N/df)'
                }
            }
        },
        'models': {
            'naive_bayes': nb_results,
            'knn': knn_results
        },
        'confidence_metrics_explanation': {
            'accuracy': 'Tỷ lệ % dự đoán đúng',
            'average_confidence': 'Giá trị confidence trung bình của model',
            'expected_calibration_error': 'ECE - đo mức độ tin cậy của confidence (càng thấp càng tốt, <5% là tốt)',
            'coverage_at_threshold': 'Tỷ lệ % mẫu có confidence >= threshold',
            'accuracy_at_threshold': 'Accuracy chỉ tính trên các mẫu có confidence >= threshold'
        }
    }
    
    return report


def save_report(report, format='both'):
    """Lưu báo cáo ra file JSON và/hoặc Markdown"""
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    if format in ['json', 'both']:
        json_path = os.path.join(RESULTS_DIR, f'evaluation_report_{timestamp}.json')
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print(f"📁 Đã lưu: {json_path}")
    
    if format in ['md', 'both']:
        md_path = os.path.join(RESULTS_DIR, f'evaluation_report_{timestamp}.md')
        md_content = generate_markdown_report(report)
        with open(md_path, 'w', encoding='utf-8') as f:
            f.write(md_content)
        print(f"📁 Đã lưu: {md_path}")
    
    return json_path if format in ['json', 'both'] else md_path


def generate_markdown_report(report):
    """Tạo báo cáo dạng Markdown với Raw vs Calibrated comparison"""
    
    nb = report['models']['naive_bayes']
    knn = report['models']['knn']
    
    md = f"""# 📊 BÁO CÁO ĐÁNH GIÁ AI CHATBOT (CALIBRATED CONFIDENCE)

> **Thời gian đánh giá:** {report['evaluation_timestamp']}  
> **Tổng số mẫu test:** {report['summary']['total_test_samples']}

---

## 🧹 1. QUY TRÌNH TIỀN XỬ LÝ (Preprocessing)

{chr(10).join('- ' + step for step in report['preprocessing_pipeline']['steps'])}

### TF-IDF Vectorization
| Tham số | Giá trị |
|---------|---------|
| max_features | {report['preprocessing_pipeline']['vectorization']['params']['max_features']} |
| ngram_range | {report['preprocessing_pipeline']['vectorization']['params']['ngram_range']} |
| sublinear_tf | {report['preprocessing_pipeline']['vectorization']['params']['sublinear_tf']} |
| **Công thức** | `{report['preprocessing_pipeline']['vectorization']['params']['formula']}` |

---

## 🤖 2. NAIVE BAYES (Phân loại Topic)

### Kỹ thuật sử dụng
- **Thuật toán:** {nb['technique']['algorithm']}
- **Công thức:** `{nb['technique']['formula']}`
- **Smoothing:** {nb['technique']['smoothing']}

### Confidence Calibration (Temperature Scaling)

| Tham số | Giá trị |
|---------|---------|
| **Method** | {nb['confidence_calibration']['method']} |
| **Formula** | `{nb['confidence_calibration']['formula']}` |
| **Temperature** | {nb['confidence_calibration']['temperature']} |
| **Ý nghĩa** | {nb['confidence_calibration']['interpretation']} |

### Kết quả (RAW vs CALIBRATED)

| Metric | Raw | Calibrated |
|--------|-----|------------|
| **Accuracy** | {nb['metrics']['raw']['accuracy']:.2f}% | (không đổi) |
| **Avg Confidence** | {nb['metrics']['raw']['average_confidence']:.2f}% | **{nb['metrics']['calibrated']['average_confidence']:.2f}%** |
| **ECE** | {nb['metrics']['raw']['expected_calibration_error']:.2f}% | **{nb['metrics']['calibrated']['expected_calibration_error']:.2f}%** |

### Coverage & Accuracy theo Threshold (Calibrated)

| Threshold | Coverage | Accuracy |
|-----------|----------|----------|
| ≥ 0.3 | {nb['metrics']['calibrated']['coverage_at_threshold'].get(0.3, 0):.1f}% | {nb['metrics']['calibrated']['accuracy_at_threshold'].get(0.3, 0):.1f}% |
| ≥ 0.5 | {nb['metrics']['calibrated']['coverage_at_threshold'].get(0.5, 0):.1f}% | {nb['metrics']['calibrated']['accuracy_at_threshold'].get(0.5, 0):.1f}% |
| ≥ 0.7 | {nb['metrics']['calibrated']['coverage_at_threshold'].get(0.7, 0):.1f}% | {nb['metrics']['calibrated']['accuracy_at_threshold'].get(0.7, 0):.1f}% |
| ≥ 0.9 | {nb['metrics']['calibrated']['coverage_at_threshold'].get(0.9, 0):.1f}% | {nb['metrics']['calibrated']['accuracy_at_threshold'].get(0.9, 0):.1f}% |

---

## 🔍 3. KNN (Tìm câu trả lời)

### Kỹ thuật sử dụng
- **Thuật toán:** {knn['technique']['algorithm']}
- **Distance Metric:** `{knn['technique']['distance_metric']}`
- **Raw Confidence:** `{knn['technique']['raw_confidence_formula']}`
- **K neighbors:** {knn['technique']['k_neighbors']}

### Preprocessing cho KNN (Khác với NB)
{chr(10).join('- ' + step for step in knn['technique']['preprocessing'])}

### Confidence Calibration (Sigmoid Scaling)

| Tham số | Giá trị |
|---------|---------|
| **Method** | {knn['confidence_calibration']['method']} |
| **Formula** | `{knn['confidence_calibration']['formula']}` |
| **k (steepness)** | {knn['confidence_calibration']['k']} |
| **midpoint** | {knn['confidence_calibration']['midpoint']} |
| **Ý nghĩa** | {knn['confidence_calibration']['interpretation']} |

### Kết quả (RAW vs CALIBRATED)

| Metric | Raw | Calibrated |
|--------|-----|------------|
| **Exact Match** | {knn['metrics']['raw']['exact_match_accuracy']:.2f}% | (không đổi) |
| **Avg Confidence** | {knn['metrics']['raw']['average_confidence']:.2f}% | **{knn['metrics']['calibrated']['average_confidence']:.2f}%** |

### Coverage & Accuracy theo Threshold (Calibrated)

| Threshold | Coverage | Accuracy |
|-----------|----------|----------|
| ≥ 0.3 | {knn['metrics']['calibrated']['coverage_at_threshold'].get(0.3, 0):.1f}% | {knn['metrics']['calibrated']['accuracy_at_threshold'].get(0.3, 0):.1f}% |
| ≥ 0.5 | {knn['metrics']['calibrated']['coverage_at_threshold'].get(0.5, 0):.1f}% | {knn['metrics']['calibrated']['accuracy_at_threshold'].get(0.5, 0):.1f}% |
| ≥ 0.7 | {knn['metrics']['calibrated']['coverage_at_threshold'].get(0.7, 0):.1f}% | {knn['metrics']['calibrated']['accuracy_at_threshold'].get(0.7, 0):.1f}% |
| ≥ 0.9 | {knn['metrics']['calibrated']['coverage_at_threshold'].get(0.9, 0):.1f}% | {knn['metrics']['calibrated']['accuracy_at_threshold'].get(0.9, 0):.1f}% |

---

## 📖 4. GIẢI THÍCH CALIBRATION

### Tại sao cần Calibration?
- **NB**: Confidence thường CAO quá (64% vs accuracy 47%) → Temperature Scaling làm "mềm"
- **KNN**: Cosine similarity thường THẤP với TF-IDF (0.2-0.6) → Sigmoid Scaling đưa về [0,1] hợp lý

### Metrics sau Calibration
| Metric | Ý nghĩa |
|--------|---------|
| **Accuracy** | Tỷ lệ % dự đoán đúng (không đổi) |
| **Avg Confidence** | Giá trị confidence trung bình ĐÃ CALIBRATE |
| **ECE** | Expected Calibration Error - lý tưởng nên ≈ 0% |
| **Coverage** | % mẫu có confidence ≥ threshold |

---

## 🧠 5. CÔNG THỨC CHI TIẾT

### NB Temperature Scaling
```
log_proba = log P(c) + Σ log P(word_i|c)
calibrated = softmax(log_proba / Temperature)
confidence = max(calibrated)
```

### KNN Sigmoid Scaling
```
raw_similarity = 1 - cosine_distance
calibrated = 1 / (1 + exp(-k × (raw_similarity - midpoint)))
```

---

*Báo cáo được tạo tự động bởi evaluate_models.py với calibrated confidence*
"""
    return md


# =========================================================
# 🚀 MAIN
# =========================================================
if __name__ == "__main__":
    print("\n" + "="*60)
    print("🚀 BẮT ĐẦU ĐÁNH GIÁ MODELS VỚI CALIBRATED CONFIDENCE")
    print("="*60)
    
    # 1. Load models và dữ liệu
    vectorizer, nb_model, knn_model = load_models()
    df_valid = load_validation_data()
    
    # 🆕 Load training data cho NB find_answer
    df_train = pd.read_csv(os.path.join(DATA_DIR, 'qa_train.csv'))
    df_train['clean_question'] = df_train['question'].apply(preprocess_text)
    print(f"📊 Đã load {len(df_train)} mẫu training cho answer lookup")
    
    # 2. Khởi tạo Calibrator
    # - NB: Temperature=1.5 (làm mềm confidence)
    # - KNN: k=10, midpoint=0.4 (sigmoid scaling)
    calibrator = UnifiedCalibrator(
        nb_temperature=1.5,
        knn_k=10.0,
        knn_midpoint=0.4
    )
    print(f"📐 Calibrator: NB(T={calibrator.nb_calibrator.temperature}), KNN(k={calibrator.knn_calibrator.k}, mid={calibrator.knn_calibrator.midpoint})")
    
    # 3. Đánh giá từng model VỚI calibration
    nb_results = evaluate_naive_bayes(nb_model, vectorizer, df_valid, df_train, calibrator)
    knn_results = evaluate_knn(knn_model, vectorizer, df_valid, calibrator)
    
    # 4. In kết quả tóm tắt
    print("\n" + "="*60)
    print("📊 KẾT QUẢ TỔNG HỢP")
    print("="*60)
    
    print(f"\n🤖 NAIVE BAYES PIPELINE:")
    print(f"   • Topic Accuracy: {nb_results['metrics']['topic_accuracy']:.2f}%")
    print(f"   • Answer Accuracy: {nb_results['metrics']['answer_accuracy']:.2f}%")
    print(f"   • Avg Answer Confidence: {nb_results['metrics']['average_answer_confidence']:.2f}%")
    
    print(f"\n🔍 KNN (Answer Retrieval):")
    print(f"   • Exact Match: {knn_results['metrics']['raw']['exact_match_accuracy']:.2f}%")
    print(f"   • Avg Confidence: {knn_results['metrics']['raw']['average_confidence']:.2f}%")
    
    # 5. Tạo và lưu báo cáo
    report = generate_report(nb_results, knn_results)
    saved_path = save_report(report, format='json')  # Chỉ lưu JSON
    
    print("\n" + "="*60)
    print("✅ ĐÁNH GIÁ HOÀN TẤT!")
    print("="*60)

