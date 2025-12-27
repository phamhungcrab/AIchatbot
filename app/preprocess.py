# -------------------------------
# 🧹 preprocess.py — Tiền xử lý văn bản Tiếng Việt tối ưu (Refactored)
# -------------------------------

import re
import pickle
from pyvi import ViTokenizer 
from sklearn.feature_extraction.text import TfidfVectorizer

# =========================================================
# 🧠 CLASS TEXT PREPROCESSOR (SINGLETON)
# =========================================================
class TextPreprocessor:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(TextPreprocessor, cls).__new__(cls)
            cls._instance._initialize()
        return cls._instance

    def _initialize(self):
        """Khởi tạo các tài nguyên, compile regex một lần duy nhất."""
        
        # 1. Compile Regex Patterns (Tối ưu tốc độ)
        self.re_special_chars = re.compile(r'[^\w\s]')
        self.re_numbers = re.compile(r'\d+')
        
        # 2. Load Data Dictionaries
        self._load_dictionaries()
        
        # 3. Build Synonym Regex (Tối ưu tìm kiếm O(1))
        # Tạo pattern dạng: \b(phrase1|phrase2|...)\b
        # Sắp xếp theo độ dài giảm dần để ưu tiên cụm từ dài trước (Longest Match)
        all_phrases = [p for p in self.SYNONYMS.keys() if " " in p]
        all_phrases.sort(key=len, reverse=True)
        if all_phrases:
            pattern = r'\b(' + '|'.join(map(re.escape, all_phrases)) + r')\b'
            self.re_synonym_phrases = re.compile(pattern, re.IGNORECASE)
        else:
            self.re_synonym_phrases = None

    def _load_dictionaries(self):
        """Định nghĩa các từ điển dữ liệu."""
        self.VIETNAMESE_STOPWORDS = {
            'thì', 'là', 'mà', 'và', 'của', 'những', 'các', 'như', 'thế', 'nào', 
            'được', 'về', 'với', 'trong', 'có', 'không', 'cho', 'tôi', 'bạn', 
            'cậu', 'tớ', 'mình', 'nó', 'hắn', 'gì', 'cái', 'con', 'người', 
            'sự', 'việc', 'đó', 'đây', 'kia', 'này', 'nhé', 'ạ', 'ơi', 'đi', 
            'làm', 'khi', 'lúc', 'nơi', 'tại', 'đã', 'đang', 'sẽ', 'muốn', 
            'phải', 'biết', 'hãy', 'rồi', 'chứ', 'nhỉ'
        }
        
        # 🆕 Từ khóa quan trọng KHÔNG được xóa khi preprocessing cho KNN
        # Sử dụng cho semantic matching - cần giữ context
        self.CRITICAL_KEYWORDS = {
            # Thuật toán AI/ML
            'knn', 'bfs', 'dfs', 'svm', 'cnn', 'rnn', 'lstm', 'transformer',
            'naive', 'bayes', 'decision', 'tree', 'random', 'forest',
            'gradient', 'descent', 'backpropagation', 'softmax', 'sigmoid',
            # Search algorithms
            'minimax', 'alpha', 'beta', 'heuristic', 'admissible', 'consistent',
            'ucs', 'ids', 'a*', 'greedy',
            # Logic
            'modus', 'ponens', 'resolution', 'cnf', 'fol', 'kb',
            # Từ khóa hỏi đáp quan trọng (giữ cho KNN)
            'là', 'gì', 'khác', 'giống', 'so', 'sánh', 'tại', 'sao', 'như', 'nào',
            # Topics
            'agent', 'tác', 'tử', 'môi', 'trường', 'học', 'máy', 'sâu'
        }
        
        # Stopwords nhẹ cho KNN - chỉ xóa các từ thực sự là noise
        self.LIGHT_STOPWORDS = {
            'thì', 'mà', 'và', 'của', 'những', 'các', 'được', 'cho', 'tôi', 'bạn',
            'cậu', 'tớ', 'mình', 'nó', 'hắn', 'cái', 'con', 'sự', 'việc',
            'đó', 'đây', 'kia', 'này', 'nhé', 'ạ', 'ơi', 'đi', 'rồi', 'chứ', 'nhỉ'
        }

        self.SYNONYMS = {
            # 1. Thuật toán & Khái niệm cơ bản
            "knn": ["k-nearest neighbors", "k nearest neighbors", "lân cận gần nhất", "k lân cận"],
            "naive bayes": ["naïve bayes", "bayes ngây thơ", "bayes"],
            "bfs": ["breadth-first search", "tìm kiếm theo chiều rộng", "chiều rộng"],
            "dfs": ["depth-first search", "tìm kiếm theo chiều sâu", "chiều sâu"],
            "a*": ["a star", "a sao", "thuật toán a*"],
            
            # 2. Logic & Suy diễn
            "logic mệnh đề": ["propositional logic", "logic phát biểu"],
            "logic vị từ": ["first-order logic", "fol", "logic bậc nhất"],
            "kb": ["knowledge base", "cơ sở tri thức"],
            
            # 3. Học máy (Machine Learning)
            "học có giám sát": ["supervised learning", "học giám sát"],
            "học không giám sát": ["unsupervised learning", "học không giám sát"],
            "học tăng cường": ["reinforcement learning", "rl"],
            "học máy": ["machine learning", "ml"],
            "trí tuệ nhân tạo": ["ai", "artificial intelligence"],
            "xử lý ngôn ngữ": ["nlp", "natural language processing"],
            
            # 4. Tác tử & Môi trường
            "tác tử": ["agent", "đại lý"],
            "peas": ["performance environment actuators sensors", "độ đo môi trường bộ chấp hành cảm biến"],
            "môi trường": ["environment"],
            "cảm biến": ["sensors"],
            "bộ chấp hành": ["actuators"],

            # 6. Từ khóa hỏi đáp thông dụng
            "là gì": ["là cái gì", "nghĩa là gì", "định nghĩa", "khái niệm", "chức năng", "tác dụng", "công dụng", "vai trò", "ý nghĩa", "dùng để làm gì"],
            "tại sao": ["vì sao", "lý do", "nguyên nhân"],
            "như thế nào": ["ra sao", "làm sao", "cách nào"],
            
            # 5. Thiết bị & Đời sống
            "xe hơi": ["ô tô", "xế hộp", "bốn bánh"],
            "điện thoại": ["dế", "smartphone", "mobile", "di động"],
            "máy tính": ["laptop", "pc", "computer", "desktop"],
            "kém": ["tệ", "xấu", "yếu", "dở"],
            "tốt": ["ngon", "xịn", "đỉnh", "tuyệt", "hay"]
        }

        self.WEIGHTED_KEYWORDS = {
            "giá": 2.0, "mua": 1.5, "bán": 1.5, "lỗi": 2.0,
            "không": 1.5, "tại sao": 1.5, "là gì": 1.2,
            # 🔥 Từ khóa so sánh - ưu tiên cao để nhận diện câu hỏi so sánh
            "khác": 3.0, "khác gì": 3.0, "khác nhau": 3.0, 
            "so sánh": 3.0, "so với": 2.5, "giống": 2.5,
            "khác biệt": 3.0, "điểm khác": 3.0
        }

        self.NEGATION_WORDS = {"không", "chẳng", "chả", "đừng", "chưa", "kém", "đâu"}

        # Tạo mapping ngược (Canonicalization)
        self.REVERSE_SYNONYMS = {}
        for canonical, variations in self.SYNONYMS.items():
            for var in variations:
                self.REVERSE_SYNONYMS[var] = canonical
            self.REVERSE_SYNONYMS[canonical] = canonical

    def preprocess_text(self, text: str) -> str:
        """Quy trình: Lowercase -> Xóa ký tự lạ -> Tách từ (PyVi) -> Lọc Stopwords"""
        if not text: return ""

        # 1. Lowercase & Clean (Dùng Compiled Regex)
        text = text.lower()
        text = self.re_special_chars.sub('', text)
        text = self.re_numbers.sub('', text)

        # 2. Tokenize (PyVi)
        tokenized_text = ViTokenizer.tokenize(text)

        # 3. Filter Stopwords
        tokens = tokenized_text.split()
        filtered_tokens = [
            word for word in tokens 
            if word not in self.VIETNAMESE_STOPWORDS and len(word) > 1
        ]

        return ' '.join(filtered_tokens)

    def preprocess_for_knn(self, text: str) -> str:
        """
        🆕 Preprocessing nhẹ cho KNN - giữ lại từ khóa quan trọng.
        
        Khác với preprocess_text (NB):
        - Dùng LIGHT_STOPWORDS thay vì VIETNAMESE_STOPWORDS 
        - Giữ lại CRITICAL_KEYWORDS (thuật ngữ AI/ML)
        - Mở rộng với synonyms để tăng matching
        
        Args:
            text: Câu hỏi gốc của user
            
        Returns:
            str: Câu đã preprocess, phù hợp cho cosine similarity
        """
        if not text: return ""

        # 1. Lowercase & Clean (giữ nguyên như preprocess_text)
        text = text.lower()
        text = self.re_special_chars.sub('', text)
        # KHÔNG xóa số cho KNN (có thể quan trọng: k=5, top-5, etc.)
        
        # 2. Tokenize (PyVi)
        tokenized_text = ViTokenizer.tokenize(text)
        
        # 3. Filter với LIGHT_STOPWORDS - giữ lại nhiều context hơn
        tokens = tokenized_text.split()
        filtered_tokens = []
        
        for word in tokens:
            # Giữ lại nếu là critical keyword HOẶC không phải light stopword
            if word in self.CRITICAL_KEYWORDS:
                filtered_tokens.append(word)  # Luôn giữ critical keywords
            elif word not in self.LIGHT_STOPWORDS and len(word) > 1:
                filtered_tokens.append(word)
        
        # 4. Mở rộng với synonyms (tăng khả năng matching)
        processed_text = ' '.join(filtered_tokens)
        expanded_text = self.expand_query(processed_text)
        
        return expanded_text

    def expand_query(self, text: str) -> str:
        """Mở rộng truy vấn bằng cách thêm từ đồng nghĩa (Optimized)."""
        if not text: return ""
        
        expanded_words = []
        text_lower = text.lower()
        
        # 1. Mở rộng từ đơn
        words = text.split()
        for word in words:
            expanded_words.append(word)
            if word.lower() in self.SYNONYMS:
                expanded_words.extend(self.SYNONYMS[word.lower()])
        
        # 2. Mở rộng cụm từ (Dùng Regex thay vì Loop)
        if self.re_synonym_phrases:
            matches = self.re_synonym_phrases.findall(text_lower)
            for match in matches:
                # match là cụm từ tìm thấy (ví dụ "xe hơi") -> lấy synonyms của nó
                if match in self.SYNONYMS:
                    expanded_words.extend(self.SYNONYMS[match])

        return " ".join(expanded_words)

    def detect_negation(self, text: str) -> str:
        """Phát hiện và xử lý phủ định."""
        if not text: return ""
        tokens = text.split()
        processed = []
        negation_active = False
        
        for token in tokens:
            if token.lower() in self.NEGATION_WORDS:
                negation_active = True
                processed.append(token)
            elif negation_active:
                processed.append(f"NOT_{token}")
                negation_active = False
            else:
                processed.append(token)
        return " ".join(processed)

    def weighted_keyword_match(self, text: str) -> float:
        """Tính điểm khớp từ khóa quan trọng."""
        if not text: return 0.0
        score = 0.0
        text_lower = text.lower()
        for kw, weight in self.WEIGHTED_KEYWORDS.items():
            if kw in text_lower:
                score += weight
        return score

    def canonicalize_text(self, text: str) -> set:
        """Chuẩn hóa văn bản về dạng từ khóa gốc."""
        if not text: return set()
        
        # Gọi preprocess_text nội bộ
        tokens = self.preprocess_text(text).split()
        canonical_tokens = set()
        
        for token in tokens:
            if token in self.REVERSE_SYNONYMS:
                canonical_tokens.add(self.REVERSE_SYNONYMS[token])
            else:
                canonical_tokens.add(token)
        return canonical_tokens

    def calculate_jaccard_similarity(self, text1: str, text2: str) -> float:
        """Tính độ tương đồng Jaccard trên tập từ đã chuẩn hóa."""
        if not text1 or not text2: return 0.0
        
        set1 = self.canonicalize_text(text1)
        set2 = self.canonicalize_text(text2)
        
        if not set1 and not set2: return 0.0
        
        intersection = set1.intersection(set2)
        union = set1.union(set2)
        
        return len(intersection) / len(union) if union else 0.0

# =========================================================
# 🚀 MODULE LEVEL INTERFACE (BACKWARD COMPATIBILITY)
# =========================================================

# Khởi tạo Singleton
preprocessor = TextPreprocessor()

# Expose các hàm để các module khác import như cũ
def preprocess_text(text: str) -> str:
    return preprocessor.preprocess_text(text)

def preprocess_for_knn(text: str) -> str:
    """🆕 Preprocessing nhẹ cho KNN - giữ từ khóa quan trọng."""
    return preprocessor.preprocess_for_knn(text)

def expand_query(text: str) -> str:
    return preprocessor.expand_query(text)

def detect_negation(text: str) -> str:
    return preprocessor.detect_negation(text)

def weighted_keyword_match(text: str) -> float:
    return preprocessor.weighted_keyword_match(text)

def calculate_jaccard_similarity(text1: str, text2: str) -> float:
    return preprocessor.calculate_jaccard_similarity(text1, text2)

def train_vectorizer(corpus):
    """Giữ nguyên hàm train_vectorizer vì nó độc lập."""
    vectorizer = TfidfVectorizer(
        max_features=800,
        ngram_range=(1, 2),
        min_df=1,
        sublinear_tf=True
    )
    vectorizer.fit(corpus)
    return vectorizer