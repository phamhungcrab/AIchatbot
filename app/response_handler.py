# -------------------------------
# 🎯 response_handler.py — Xử lý response theo độ tin cậy
# 
# Quy tắc:
#   - Confidence ≥ 80%: Trả lời bình thường
#   - 50% ≤ Confidence < 80%: Trả lời kèm cảnh báo độ tin cậy
#   - Confidence < 50%: Không trả lời, xin diễn đạt lại
# -------------------------------

from confidence_utils import UnifiedCalibrator


class ConfidenceResponseHandler:
    """
    Xử lý response dựa trên confidence level.
    
    📌 Cách hoạt động:
    1. Nhận confidence (đã calibrate) từ model
    2. Phân loại theo threshold
    3. Format response phù hợp
    
    📌 Thresholds:
    - HIGH_CONFIDENCE = 0.8 (80%)
    - MIN_CONFIDENCE = 0.5 (50%)
    """
    
    def __init__(self, high_threshold=0.8, min_threshold=0.5):
        """
        Args:
            high_threshold: Ngưỡng trả lời tự tin (mặc định 80%)
            min_threshold: Ngưỡng tối thiểu để trả lời (mặc định 50%)
        """
        self.high_threshold = high_threshold
        self.min_threshold = min_threshold
    
    def format_response(self, answer, confidence, topic=None):
        """
        Format response dựa trên confidence level.
        
        Args:
            answer: Câu trả lời từ model
            confidence: Độ tin cậy (0.0 - 1.0, đã calibrate)
            topic: Topic dự đoán (optional)
            
        Returns:
            dict: {
                'response': str (câu trả lời đã format),
                'confidence': float,
                'level': str ('high', 'medium', 'low'),
                'should_answer': bool
            }
        """
        # Chuyển confidence về % để dễ đọc
        conf_percent = confidence * 100
        
        # CASE 1: Confidence CAO (≥ 80%) → Trả lời bình thường
        if confidence >= self.high_threshold:
            return {
                'response': answer,
                'confidence': confidence,
                'confidence_display': f"{conf_percent:.0f}%",
                'level': 'high',
                'level_emoji': '🟢',
                'should_answer': True
            }
        
        # CASE 2: Confidence TRUNG BÌNH (50% - 80%) → Trả lời kèm cảnh báo
        elif confidence >= self.min_threshold:
            warning_response = (
                f"⚠️ _[Độ tin cậy: {conf_percent:.0f}%]_\n\n"
                f"{answer}"
            )
            return {
                'response': warning_response,
                'confidence': confidence,
                'confidence_display': f"{conf_percent:.0f}%",
                'level': 'medium',
                'level_emoji': '🟡',
                'should_answer': True
            }
        
        # CASE 3: Confidence THẤP (< 50%) → Không trả lời
        else:
            fallback_response = (
                "🤔 Xin lỗi, tôi không chắc chắn về câu trả lời này.\n"
                "Bạn có thể diễn đạt lại câu hỏi được không?"
            )
            return {
                'response': fallback_response,
                'confidence': confidence,
                'confidence_display': f"{conf_percent:.0f}%",
                'level': 'low',
                'level_emoji': '🔴',
                'should_answer': False
            }
    
    def get_level_description(self, level):
        """Mô tả chi tiết cho từng level."""
        descriptions = {
            'high': "Độ tin cậy cao - Trả lời tự tin",
            'medium': "Độ tin cậy trung bình - Trả lời kèm cảnh báo",
            'low': "Độ tin cậy thấp - Không đủ tự tin để trả lời"
        }
        return descriptions.get(level, "Unknown")


# =========================================================
# 🚀 MODULE-LEVEL INTERFACE (Dễ dùng)
# =========================================================

# Default handler với threshold chuẩn
_default_handler = ConfidenceResponseHandler(
    high_threshold=0.8,
    min_threshold=0.5
)

def format_chatbot_response(answer, confidence, topic=None):
    """
    Interface đơn giản để format response.
    
    Ví dụ:
        result = format_chatbot_response("KNN là...", 0.75)
        print(result['response'])  # Có cảnh báo vì < 80%
    """
    return _default_handler.format_response(answer, confidence, topic)


# =========================================================
# 🧪 SANITY CHECK
# =========================================================
if __name__ == "__main__":
    print("="*60)
    print("🧪 TEST CONFIDENCE RESPONSE HANDLER")
    print("="*60)
    
    handler = ConfidenceResponseHandler(
        high_threshold=0.8,
        min_threshold=0.5
    )
    
    test_cases = [
        ("KNN là thuật toán phân loại dựa trên k láng giềng gần nhất.", 0.92),
        ("Naive Bayes sử dụng định lý Bayes để phân loại.", 0.65),
        ("BFS duyệt theo chiều rộng.", 0.35),
    ]
    
    print(f"\n📐 Thresholds: HIGH ≥ {handler.high_threshold*100:.0f}%, MIN ≥ {handler.min_threshold*100:.0f}%\n")
    
    for answer, conf in test_cases:
        result = handler.format_response(answer, conf)
        print(f"{result['level_emoji']} Confidence: {result['confidence_display']} ({result['level'].upper()})")
        print(f"   Response: {result['response'][:60]}...")
        print(f"   Should answer: {result['should_answer']}")
        print()
    
    print("✅ Sanity check passed!")
