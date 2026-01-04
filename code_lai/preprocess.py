#preprocess.py - Tiền xử lý văn bản Tiếng Việt

# Phase 1: Imports
# TODO: import re, pyvi, sklearn
import re
from pyvi import ViTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer


# Phase 2: Class TextPreprocessor (Singleton)
# TODO: Tạo class với __new__ và _initialize

class TextPreprocessor:
    _instance = None
    def __new__ (cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialize()
        return cls._instance
        
    def _initialize(self):
        self.re_special_chars = re.compile(r'[^\w\s]')
        self.re_numbers = re.compile(r'\d+')

        self._load_dictionaries()
        ##Làm để không phải duyệt lại từ đầu mỗi lần tìm kiếm từ so sánh
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

        self.REVERSE_SYNONYMS = {}
        ## TÌm mấy cái từ REVERSE truy ngược lại key
        for canonical, variations in self.SYNONYMS.items():
            for var in variations:
                self.REVERSE_SYNONYMS[var] = canonical
            self.REVERSE_SYNONYMS[canonical] = canonical

    def preprocess_text(self, text: str) -> str:
        if not text: return ""

        text = text.lower()
        text = self.re_special_chars.sub('', text)
        text = self.re_numbers.sub("", text)
        
        text_tokenized = ViTokenizer.tokenize(text)

        tokens = text_tokenized.split()

        filtered = [token for token in tokens if token not in self.VIETNAMESE_STOPWORDS and len(token) > 1]
    
        return " ".join(filtered)

    def preprocess_knn(self, text: str) -> str:
        if not text: return ""

        text = text.lower()
        text = self.re_special_chars.sub("", text)

        tokenized_text = ViTokenizer.tokenize(text)
        
        tokens = tokenized_text.split()

        filtered = [token for token in tokens if token in self.CRITICAL_KEYWORDS or (token not in self.LIGHT_STOPWORDS and len(token) > 1)]

        expended_text = self.expand_query(" ".join(filtered))
        return expended_text

    def expand_query(self, query) -> str:
        if not query: return ""

        text = query.lower()
        expanded_tokens = []
        words = text.split()

        for word in words:
            expanded_tokens.append(word)
            if word in self.SYNONYMS:
                expanded_tokens.extend(self.SYNONYMS[word])
        
        
        if self.re_synonym_phrases:
            matches = self.re_synonym_phrases.findall(text)
            for match in matches:
                if match.lower() in self.SYNONYMS:
                    expanded_tokens.extend(self.SYNONYMS[match.lower()])
                    
        return " ".join(expanded_tokens)
            
    def detect_negation(self, text: str) -> str:
        if not text: return ""

        tokens = text.split()
        negation = False
        processed = []

        for token in tokens:
            if token.lower() in self.NEGATION_WORDS:
                negation = True
                processed.append(token)
            elif negation:
                processed.append(f"NOT_{token}")
            else:
                processed.append(token)
            
        return " ".join(processed)

    def weighted_keywords(self, text: str) -> float:

        if not text : return 0.0
        score = 0.0
        text_lower = text.lower()

        for kw, weight in self.WEIGHTED_KEYWORDS.items():
            if kw in text_lower:
                score += weight
        return score
        
    def canonicalize_text(self, text: str) -> set:
        if not text: return ""

        text_lower = text.lower()
        canonical_tokens = set()

        if self.re_synonym_phrases:
            matches = self.re_synonym_phrases.findall(text_lower)
            for match in matches:
                if match.lower() in self.REVERSE_SYNONYMS:
                    canonical_tokens.add(self.REVERSE_SYNONYMS[match.lower()])
        
        for token in text_lower.split():
            if token in self.REVERSE_SYNONYMS:
                canonical_tokens.add(self.REVERSE_SYNONYMS[token])

        return canonical_tokens



    def calculate_jaccard_similarity(self, text1: str, text2: str) -> float:
        if not text1 or not text2: return 0.0

        set1 = self.canonicalize_text(text1)
        set2 = self.canonicalize_text(text2)

        if not set1 and not set2: return 0.0

        intersection = set1.intersection(set2)
        union = set1.union(set2)

        return len(intersection) / len(union) if union else 0.0
# ============================================
# Phase 6: Module-Level Interface
# ============================================
preprocessor = TextPreprocessor()

def preprocess_text(text): return preprocessor.preprocess_text(text)
def preprocess_knn(text): return preprocessor.preprocess_knn(text)
def expand_query(text): return preprocessor.expand_query(text)
def detect_negation(text): return preprocessor.detect_negation(text)
def weighted_keywords(text): return preprocessor.weighted_keywords(text)
def calculate_jaccard_similarity(text1, text2): return preprocessor.calculate_jaccard_similarity(text1, text2)
def canonicalize_text(text): return preprocessor.canonicalize_text(text)

def train_vectorizer(corpus):
    vectorizer = TfidfVectorizer(max_features=800, ngram_range=(1, 2))
    vectorizer.fit(corpus)
    return vectorizer

# ============================================
# Phase 7: Sanity Check
# ============================================
if __name__ == "__main__":
    print("🧪 Testing preprocess.py...\n")
    
    # Test 1: preprocess_text (Naive Bayes)
    test1 = "Học máy là gì vậy bạn?"
    result1 = preprocess_text(test1)
    print(f"✅ preprocess_text:")
    print(f"   Input:  '{test1}'")
    print(f"   Output: '{result1}'\n")
    
    # Test 2: preprocess_knn
    test2 = "AI và KNN khác nhau như thế nào?"
    result2 = preprocess_knn(test2)
    print(f"✅ preprocess_knn:")
    print(f"   Input:  '{test2}'")
    print(f"   Output: '{result2}'\n")
    
    # Test 3: expand_query
    test3 = "trí tuệ nhân tạo"
    result3 = expand_query(test3)
    print(f"✅ expand_query:")
    print(f"   Input:  '{test3}'")
    print(f"   Output: '{result3}'\n")
    
    # Test 4: detect_negation
    test4 = "Học máy không tốt"
    result4 = detect_negation(test4)
    print(f"✅ detect_negation:")
    print(f"   Input:  '{test4}'")
    print(f"   Output: '{result4}'\n")
    
    # Test 5: weighted_keyword_match
    test5 = "Tại sao KNN lại khác Naive Bayes?"
    result5 = preprocessor.weighted_keywords(test5)
    print(f"✅ weighted_keyword_match:")
    print(f"   Input:  '{test5}'")
    print(f"   Score: {result5}\n")
    
    # Test 6: calculate_jaccard_similarity
    text_a = "Học máy là gì"
    text_b = "Machine learning là cái gì"
    result6 = preprocessor.calculate_jaccard_similarity(text_a, text_b)
    print(f"✅ calculate_jaccard_similarity:")
    print(f"   Text A: '{text_a}'")
    print(f"   Text B: '{text_b}'")
    print(f"   Similarity: {result6:.2f}\n")

    text6 = "ml và machine learning là một và là một nhánh của AI và ai và trí tuệ nhân tạo"
    result6 = canonicalize_text(text6)
    print(f"✅ canonicalize_text:")
    print(f"   Input:  '{text6}'")
    print(f"   Output: '{result6}'\n")
    
    print("🎉 All tests completed!")