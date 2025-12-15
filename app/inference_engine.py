import re
import datetime
import math

class RuleBasedEngine:
    """
    Mô hình suy luận dựa trên luật (Rule-Based Inference Engine).
    Đóng vai trò là 'fx model' để tính toán/suy luận câu trả lời 
    mà không cần gọi LLM hay tra cứu database.
    """
    
    def __init__(self):
        self.rules = [
            (r'tính\s+([\d\.]+)\s*([\+\-\*\/])\s*([\d\.]+)', self._handle_math),
            (r'mấy giờ rồi|thời gian hiện tại', self._handle_time),
            (r'hôm nay là ngày mấy|ngày bao nhiêu', self._handle_date),
            (r'căn bậc hai của\s+([\d\.]+)', self._handle_sqrt),
            (r'bình phương của\s+([\d\.]+)', self._handle_square),
        ]

    def infer(self, text):
        """
        Duyệt qua các luật để tìm câu trả lời.
        Trả về (answer, confidence) hoặc (None, 0.0)
        """
        text_lower = text.lower()
        
        for pattern, handler in self.rules:
            match = re.search(pattern, text_lower)
            if match:
                return handler(match), 1.0  # Confidence 1.0 vì là logic toán học
                
        return None, 0.0

    def _handle_math(self, match):
        try:
            a = float(match.group(1))
            operator = match.group(2)
            b = float(match.group(3))
            
            if operator == '+': result = a + b
            elif operator == '-': result = a - b
            elif operator == '*': result = a * b
            elif operator == '/': 
                if b == 0: return "Không thể chia cho 0."
                result = a / b
                
            return f"Kết quả là: {result}"
        except:
            return "Xin lỗi, tôi không tính được phép toán này."

    def _handle_time(self, match):
        now = datetime.datetime.now()
        return f"Bây giờ là {now.strftime('%H:%M')}."

    def _handle_date(self, match):
        now = datetime.datetime.now()
        return f"Hôm nay là ngày {now.strftime('%d/%m/%Y')}."

    def _handle_sqrt(self, match):
        try:
            val = float(match.group(1))
            return f"Căn bậc hai của {val} là {math.sqrt(val):.2f}"
        except:
            return "Lỗi tính toán."

    def _handle_square(self, match):
        try:
            val = float(match.group(1))
            return f"Bình phương của {val} là {val**2}"
        except:
            return "Lỗi tính toán."

# Singleton instance
inference_engine = RuleBasedEngine()
