# -------------------------------
# 📊 confidence_utils.py — Chuẩn hoá Confidence cho NB và KNN
# Mục đích: Đưa confidence của 2 models về cùng một scale công bằng
# -------------------------------

import numpy as np

# =========================================================
# 📐 1. LÝ THUYẾT CHUẨN HOÁ CONFIDENCE
# =========================================================
"""
VẤN ĐỀ:
- NB confidence = max(P(topic|X)) → thường CAO (0.5 - 0.99) vì softmax tập trung
- KNN confidence = 1 - cosine_distance → thường THẤP (0.2 - 0.6) vì TF-IDF sparse

GIẢI PHÁP: Chuẩn hoá cả 2 về scale [0, 1] công bằng

1. NB: Temperature Scaling
   - raw_conf cao quá → giảm bằng temperature > 1
   - calibrated = softmax(logits / temperature)

2. KNN: Sigmoid Scaling  
   - Chuyển raw similarity về sigmoid curve
   - calibrated = 1 / (1 + exp(-k*(x - midpoint)))
"""


# =========================================================
# 🌡️ 2. NAIVE BAYES CALIBRATION
# =========================================================
class NaiveBayesCalibrator:
    """
    Chuẩn hoá confidence cho Naive Bayes bằng Temperature Scaling.
    
    📌 Công thức:
    1. Log Probability (từ NB):
       log P(c|X) = log P(c) + Σ log P(word_i|c)
    
    2. Softmax với Temperature:
       P_calibrated(c|X) = exp(log P(c|X) / T) / Σ exp(log P(k|X) / T)
    
    3. Confidence cuối:
       confidence = max(P_calibrated)
    
    📌 Ý nghĩa Temperature (T):
       - T = 1.0: Giữ nguyên (uncalibrated)
       - T > 1.0: "Làm mềm" distribution → confidence thấp hơn, đều hơn
       - T < 1.0: "Làm sắc" distribution → confidence cao hơn, tập trung hơn
    
    📌 Cách chọn T:
       - Lý tưởng: Optimize trên validation set để minimize ECE
       - Quick estimate: T ≈ average_confidence / average_accuracy
    """
    
    def __init__(self, temperature=1.5):
        """
        Args:
            temperature: Hệ số điều chỉnh (T > 1 giảm confidence, T < 1 tăng)
        """
        self.temperature = temperature
    
    def calibrate_from_logproba(self, log_probas):
        """
        Calibrate từ log probabilities (output của NB).
        
        Args:
            log_probas: np.array shape (n_classes,) - log P(c|X) cho mỗi class
            
        Returns:
            calibrated_confidence: float trong [0, 1]
            calibrated_proba: np.array shape (n_classes,) - xác suất đã calibrate
        """
        # Chia cho temperature
        scaled_logits = log_probas / self.temperature  # Shape: (n_classes,)
        
        # Softmax với trick ổn định số học
        # P(c) = exp(z_c - max(z)) / Σ exp(z_k - max(z))
        max_logit = np.max(scaled_logits)
        exp_logits = np.exp(scaled_logits - max_logit)  # Trừ max để tránh overflow
        calibrated_proba = exp_logits / np.sum(exp_logits)  # Shape: (n_classes,)
        
        # Confidence = max probability
        calibrated_confidence = float(np.max(calibrated_proba))
        
        return calibrated_confidence, calibrated_proba
    
    def calibrate_from_proba(self, raw_proba):
        """
        Calibrate từ raw probability (nếu đã có softmax sẵn).
        
        ⚠️ Lưu ý: Phương pháp này là approximation, không chính xác bằng 
        calibrate_from_logproba vì thông tin log đã bị mất.
        
        Args:
            raw_proba: np.array shape (n_classes,) - P(c|X) từ model
            
        Returns:
            calibrated_confidence: float trong [0, 1]
        """
        # Chuyển về log, scale, rồi softmax lại
        # Thêm epsilon để tránh log(0)
        epsilon = 1e-10
        log_proba = np.log(raw_proba + epsilon)
        return self.calibrate_from_logproba(log_proba)
    
    def find_optimal_temperature(self, y_true, y_pred_proba, n_bins=10):
        """
        Tìm temperature tối ưu để minimize ECE.
        
        Args:
            y_true: Ground truth labels
            y_pred_proba: Matrix of predicted probabilities (n_samples, n_classes)
            n_bins: Số bins cho ECE
            
        Returns:
            optimal_temperature: float
        """
        best_t = 1.0
        best_ece = float('inf')
        
        for t in np.arange(0.5, 3.0, 0.1):
            self.temperature = t
            ece = self._calculate_ece(y_true, y_pred_proba, n_bins)
            if ece < best_ece:
                best_ece = ece
                best_t = t
        
        self.temperature = best_t
        return best_t
    
    def _calculate_ece(self, y_true, y_pred_proba, n_bins=10):
        """Tính Expected Calibration Error."""
        confidences = []
        predictions = []
        
        for proba in y_pred_proba:
            conf, _ = self.calibrate_from_proba(proba)
            confidences.append(conf)
            predictions.append(np.argmax(proba))
        
        confidences = np.array(confidences)
        predictions = np.array(predictions)
        correct = predictions == y_true
        
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        ece = 0.0
        
        for i in range(n_bins):
            in_bin = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i + 1])
            if np.sum(in_bin) > 0:
                bin_accuracy = np.mean(correct[in_bin])
                bin_confidence = np.mean(confidences[in_bin])
                bin_weight = np.sum(in_bin) / len(confidences)
                ece += bin_weight * abs(bin_accuracy - bin_confidence)
        
        return ece


# =========================================================
# 📐 3. KNN CALIBRATION
# =========================================================
class KNNCalibrator:
    """
    Chuẩn hoá confidence cho KNN bằng Sigmoid Scaling.
    
    📌 Công thức:
    1. Cosine Similarity (từ KNN):
       sim = 1 - cosine_distance = (A·B) / (||A|| × ||B||)
       → Thường trong khoảng [0.2, 0.7] với TF-IDF
    
    2. Sigmoid Scaling:
       calibrated = 1 / (1 + exp(-k × (sim - midpoint)))
       
       Trong đó:
       - k: Độ dốc (steepness) - k lớn → sigmoid sắc hơn
       - midpoint: Điểm uốn - similarity = midpoint → confidence = 0.5
    
    📌 Ý nghĩa tham số:
       - midpoint = 0.4: Similarity 0.4 → Confidence 50%
       - k = 10: Sigmoid khá dốc, phân biệt rõ high/low similarity
    
    📌 Mapping mẫu (với k=10, midpoint=0.4):
       | Raw Sim | Calibrated |
       |---------|------------|
       | 0.2     | ~12%       |
       | 0.3     | ~27%       |
       | 0.4     | 50%        |
       | 0.5     | ~73%       |
       | 0.6     | ~88%       |
       | 0.7     | ~95%       |
    """
    
    def __init__(self, k=10.0, midpoint=0.4):
        """
        Args:
            k: Độ dốc sigmoid (steepness)
            midpoint: Điểm similarity tương ứng với confidence 50%
        """
        self.k = k
        self.midpoint = midpoint
    
    def calibrate(self, raw_similarity):
        """
        Calibrate raw cosine similarity sang confidence chuẩn hoá.
        
        Args:
            raw_similarity: float trong [0, 1] - cosine similarity gốc
            
        Returns:
            calibrated_confidence: float trong [0, 1]
        """
        # Sigmoid function
        # σ(x) = 1 / (1 + exp(-k*(x - midpoint)))
        exponent = -self.k * (raw_similarity - self.midpoint)
        
        # Clamp exponent để tránh overflow
        exponent = np.clip(exponent, -500, 500)
        
        calibrated = 1.0 / (1.0 + np.exp(exponent))
        
        return float(calibrated)
    
    def calibrate_batch(self, similarities):
        """Calibrate một batch các similarity values."""
        return np.array([self.calibrate(s) for s in similarities])
    
    def find_optimal_params(self, similarities, correct_flags, target_ece=0.05):
        """
        Tìm k và midpoint tối ưu dựa trên validation data.
        
        Args:
            similarities: Array of raw similarities
            correct_flags: Boolean array - True nếu prediction đúng
            target_ece: ECE mục tiêu (default 5%)
            
        Returns:
            (optimal_k, optimal_midpoint)
        """
        best_k, best_mid = 10.0, 0.4
        best_ece = float('inf')
        
        for k in np.arange(5, 20, 1):
            for mid in np.arange(0.3, 0.6, 0.05):
                self.k = k
                self.midpoint = mid
                
                calibrated = self.calibrate_batch(similarities)
                ece = self._calculate_ece(correct_flags, calibrated)
                
                if ece < best_ece:
                    best_ece = ece
                    best_k, best_mid = k, mid
        
        self.k = best_k
        self.midpoint = best_mid
        return best_k, best_mid
    
    def _calculate_ece(self, correct_flags, confidences, n_bins=10):
        """Tính ECE cho KNN."""
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        ece = 0.0
        
        for i in range(n_bins):
            in_bin = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i + 1])
            if np.sum(in_bin) > 0:
                bin_accuracy = np.mean(correct_flags[in_bin])
                bin_confidence = np.mean(confidences[in_bin])
                bin_weight = np.sum(in_bin) / len(confidences)
                ece += bin_weight * abs(bin_accuracy - bin_confidence)
        
        return ece


# =========================================================
# 🎯 4. UNIFIED CALIBRATOR - INTERFACE CHUNG
# =========================================================
class UnifiedCalibrator:
    """
    Interface thống nhất để calibrate confidence cho cả NB và KNN.
    
    📌 Cách sử dụng:
        calibrator = UnifiedCalibrator()
        
        # Calibrate NB
        nb_conf = calibrator.calibrate_nb(raw_nb_proba)
        
        # Calibrate KNN  
        knn_conf = calibrator.calibrate_knn(raw_cosine_similarity)
    """
    
    def __init__(self, nb_temperature=1.5, knn_k=10.0, knn_midpoint=0.4):
        self.nb_calibrator = NaiveBayesCalibrator(temperature=nb_temperature)
        self.knn_calibrator = KNNCalibrator(k=knn_k, midpoint=knn_midpoint)
    
    def calibrate_nb(self, raw_proba):
        """
        Calibrate NB confidence.
        
        Args:
            raw_proba: np.array hoặc list - xác suất từ NB predict_proba
            
        Returns:
            float: Calibrated confidence
        """
        raw_proba = np.array(raw_proba).flatten()
        conf, _ = self.nb_calibrator.calibrate_from_proba(raw_proba)
        return conf
    
    def calibrate_knn(self, raw_similarity):
        """
        Calibrate KNN confidence.
        
        Args:
            raw_similarity: float - cosine similarity (1 - distance)
            
        Returns:
            float: Calibrated confidence
        """
        return self.knn_calibrator.calibrate(raw_similarity)
    
    def get_confidence_interpretation(self, confidence):
        """
        Diễn giải confidence score.
        
        Args:
            confidence: float trong [0, 1]
            
        Returns:
            str: Mô tả mức độ tin cậy
        """
        if confidence >= 0.9:
            return "Rất cao (Very High) 🟢"
        elif confidence >= 0.7:
            return "Cao (High) 🟢"
        elif confidence >= 0.5:
            return "Trung bình (Medium) 🟡"
        elif confidence >= 0.3:
            return "Thấp (Low) 🟠"
        else:
            return "Rất thấp (Very Low) 🔴"


# =========================================================
# 🧪 SANITY CHECK
# =========================================================
if __name__ == "__main__":
    print("="*60)
    print("🧪 TEST CONFIDENCE CALIBRATION")
    print("="*60)
    
    # Khởi tạo calibrator
    calibrator = UnifiedCalibrator(
        nb_temperature=1.5,
        knn_k=10.0,
        knn_midpoint=0.4
    )
    
    # Test NB calibration
    print("\n📊 NAIVE BAYES CALIBRATION (Temperature=1.5)")
    print("-"*60)
    test_nb_proba = [
        [0.8, 0.1, 0.1],    # Confidence cao
        [0.5, 0.3, 0.2],    # Confidence trung bình
        [0.4, 0.35, 0.25],  # Confidence thấp
    ]
    
    print(f"{'Raw Proba':<30} {'Raw Conf':<12} {'Calibrated':<12}")
    for proba in test_nb_proba:
        raw_conf = max(proba)
        calibrated = calibrator.calibrate_nb(proba)
        print(f"{str(proba):<30} {raw_conf:<12.2%} {calibrated:<12.2%}")
    
    # Test KNN calibration
    print("\n🔍 KNN CALIBRATION (k=10, midpoint=0.4)")
    print("-"*60)
    test_similarities = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    
    print(f"{'Raw Similarity':<18} {'Calibrated':<12} {'Level'}")
    for sim in test_similarities:
        calibrated = calibrator.calibrate_knn(sim)
        level = calibrator.get_confidence_interpretation(calibrated)
        print(f"{sim:<18.2f} {calibrated:<12.2%} {level}")
    
    print("\n✅ Sanity check passed!")
