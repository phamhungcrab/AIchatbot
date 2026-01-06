# -*- coding: utf-8 -*-
"""
Generate ~2000 Vietnamese theoretical Q&A with strong paraphrasing.
Outputs:
- init.sql (SQLite)
- qa_2000.csv
- qa_2000.jsonl
- chatbot_ai.db
"""

import argparse
import csv
import json
import os
import random
import re
import sqlite3
from typing import List, Tuple

random.seed(3160)

# ----------------------------
# 1) CANONICAL (base) Q&A
#    Tập trung lý thuyết, câu hỏi ngắn.
# ----------------------------
BASE_ITEMS: List[Tuple[str, str, str]] = [
    # ===== AI INTRO =====
    ("Trí tuệ nhân tạo (AI) là gì?", "AI là lĩnh vực nghiên cứu và xây dựng hệ thống có khả năng nhận biết, suy luận, học hỏi và hành động để đạt mục tiêu.", "AIIntro"),
    ("Có những cách nhìn nào về AI?", "Bốn cách nhìn: nghĩ như người, hành động như người, nghĩ hợp lý, hành động hợp lý.", "AIIntro"),
    ("Turing Test là gì?", "Turing Test đánh giá hành vi thông minh qua đối thoại: nếu khó phân biệt máy với người, máy thể hiện hành vi thông minh.", "AIIntro"),
    ("Hành động hợp lý khác gì bắt chước con người?", "Hành động hợp lý chọn hành động tốt nhất theo mục tiêu và thông tin; bắt chước con người mô phỏng hành vi người (có thể cả sai sót).", "AIIntro"),
    ("AI liên quan những lĩnh vực nào?", "AI liên quan: biểu diễn tri thức, suy luận, học máy, thị giác máy tính, xử lý ngôn ngữ tự nhiên, robot, tối ưu, xác suất…", "AIIntro"),

    # ===== AGENTS =====
    ("Tác tử (agent) là gì?", "Tác tử là thực thể cảm nhận môi trường qua cảm biến và tác động lại bằng hành động qua bộ chấp hành.", "Agents"),
    ("Hàm tác tử (agent function) là gì?", "Hàm tác tử ánh xạ từ lịch sử cảm nhận (percept sequence) sang hành động.", "Agents"),
    ("Chuỗi cảm nhận (percept sequence) là gì?", "Chuỗi cảm nhận là toàn bộ lịch sử cảm nhận của tác tử đến thời điểm hiện tại.", "Agents"),
    ("Tác tử hợp lý (rational agent) là gì?", "Tác tử hợp lý chọn hành động tối đa hóa hiệu năng kỳ vọng dựa trên cảm nhận và tri thức hiện có.", "Agents"),
    ("Tính hợp lý phụ thuộc vào gì?", "Phụ thuộc vào: tiêu chí hiệu năng, tri thức sẵn có, chuỗi cảm nhận, và tập hành động cho phép.", "Agents"),
    ("PEAS gồm những thành phần nào?", "PEAS: Performance measure, Environment, Actuators, Sensors.", "Agents"),
    ("Môi trường quan sát hoàn toàn là gì?", "Môi trường quan sát hoàn toàn là môi trường mà cảm biến cho đủ thông tin trạng thái hiện tại liên quan quyết định.", "Agents"),
    ("Môi trường ngẫu nhiên (stochastic) là gì?", "Môi trường ngẫu nhiên có kết quả hành động không chắc chắn do yếu tố khó dự đoán.", "Agents"),
    ("Tác tử phản xạ đơn giản là gì?", "Tác tử phản xạ đơn giản dùng luật if–then dựa trên cảm nhận hiện tại, không dùng bộ nhớ.", "Agents"),
    ("Tác tử dựa trên mô hình là gì?", "Tác tử dựa trên mô hình duy trì trạng thái nội bộ để suy ra trạng thái thế giới khi quan sát không đầy đủ.", "Agents"),
    ("Tác tử dựa trên mục tiêu là gì?", "Tác tử dựa trên mục tiêu chọn hành động để đạt mục tiêu, cần xét hậu quả hành động.", "Agents"),
    ("Tác tử dựa trên tiện ích là gì?", "Tác tử dựa trên tiện ích dùng hàm tiện ích để so sánh trạng thái và chọn hành động tối đa hóa tiện ích.", "Agents"),
    ("Tác tử tự trị là gì?", "Tác tử tự trị điều chỉnh hành vi dựa trên kinh nghiệm/tri thức của chính nó, ít phụ thuộc điều khiển ngoài.", "Agents"),

    # ===== SEARCH =====
    ("Bài toán tìm kiếm trong AI là gì?", "Là tìm chuỗi hành động/đường đi từ trạng thái đầu đến đích trong không gian trạng thái, thường tối ưu theo chi phí.", "Search"),
    ("Một bài toán tìm kiếm gồm những thành phần nào?", "Gồm: trạng thái ban đầu, hàm kế nhiệm (chuyển trạng thái), kiểm tra đích, và chi phí đường đi.", "Search"),
    ("Hàm kế nhiệm (successor function) là gì?", "Trả về các hành động hợp lệ từ một trạng thái và trạng thái kết quả tương ứng.", "Search"),
    ("Kiểm tra đích (goal test) là gì?", "Điều kiện xác định trạng thái có thỏa mục tiêu hay không.", "Search"),
    ("Chi phí đường đi (path cost) là gì?", "Tổng chi phí các bước từ gốc đến một nút theo hàm chi phí bước.", "Search"),
    ("BFS là gì?", "BFS mở rộng theo từng lớp độ sâu; dùng hàng đợi; tối ưu theo số bước khi chi phí bước như nhau.", "Search"),
    ("DFS là gì?", "DFS đi sâu theo một nhánh trước; dùng ngăn xếp/đệ quy; ít bộ nhớ nhưng không đảm bảo tối ưu.", "Search"),
    ("Uniform Cost Search là gì?", "UCS mở rộng nút có chi phí g(n) nhỏ nhất; tối ưu nếu chi phí bước không âm.", "Search"),
    ("Depth-Limited Search là gì?", "DFS với giới hạn độ sâu để tránh đi vô hạn; có thể bỏ lỡ nghiệm nếu giới hạn thấp.", "Search"),
    ("Iterative Deepening DFS là gì?", "IDDFS lặp DLS với giới hạn tăng dần; đầy đủ như BFS và bộ nhớ thấp như DFS.", "Search"),
    ("Tree search khác graph search thế nào?", "Tree search có thể mở rộng trùng trạng thái; graph search dùng closed set để tránh mở rộng lặp.", "Search"),
    ("Vì sao cần xử lý trạng thái lặp?", "Trạng thái lặp gây nổ số nút và vòng lặp, làm tìm kiếm kém hiệu quả.", "Search"),
    ("Heuristic h(n) là gì?", "Heuristic là ước lượng chi phí còn lại từ n đến đích để định hướng tìm kiếm.", "Search"),
    ("Tìm kiếm có thông tin là gì?", "Là tìm kiếm dùng heuristic/tri thức bổ sung ngoài mô tả bài toán để ưu tiên nút hứa hẹn.", "Search"),
    ("Greedy Best-First Search là gì?", "Chọn mở rộng nút có h(n) nhỏ nhất; nhanh nhưng không đảm bảo tối ưu.", "Search"),
    ("A* là gì?", "A* dùng f(n)=g(n)+h(n) để chọn nút; có thể tối ưu nếu heuristic phù hợp.", "Search"),
    ("Heuristic admissible là gì?", "Heuristic không bao giờ vượt quá chi phí tối ưu thực: h(n) ≤ h*(n).", "Search"),
    ("Local search là gì?", "Tìm kiếm cục bộ cải thiện dần một trạng thái hiện tại, không lưu toàn bộ cây; hợp không gian rất lớn.", "Search"),
    ("Hill Climbing là gì?", "Luôn chọn bước cải thiện; dễ kẹt cực trị địa phương hoặc cao nguyên.", "Search"),
    ("Simulated Annealing là gì?", "Cho phép đôi khi chấp nhận bước xấu với xác suất giảm dần để thoát kẹt.", "Search"),
    ("Beam Search là gì?", "Giữ lại k trạng thái tốt nhất ở mỗi bước mở rộng để giới hạn bộ nhớ.", "Search"),
    ("Genetic Algorithm là gì?", "Tối ưu dựa trên chọn lọc, lai ghép, đột biến trong quần thể nghiệm.", "Search"),
    ("IDA* là gì?", "Iterative Deepening A* dùng ngưỡng theo f=g+h thay vì độ sâu, giảm bộ nhớ so với A*.", "Search"),
    ("RBFS là gì?", "Recursive Best-First Search mô phỏng best-first với bộ nhớ tuyến tính bằng f_limit.", "Search"),
    ("SMA* là gì?", "Memory-bounded A* giới hạn bộ nhớ; loại bỏ lá tệ nhất khi đầy.", "Search"),

    # ===== ADVERSARIAL =====
    ("Tìm kiếm đối kháng là gì?", "Áp dụng cho môi trường cạnh tranh (thường 2 người chơi) nơi đối thủ ảnh hưởng trực tiếp kết quả.", "Adversarial"),
    ("Minimax là gì?", "Thuật toán chọn nước đi tối ưu trong trò chơi zero-sum, giả định đối thủ chơi tối ưu.", "Adversarial"),
    ("Alpha-Beta pruning là gì?", "Kỹ thuật cắt tỉa trong minimax giúp bỏ nhánh chắc chắn không ảnh hưởng quyết định tối ưu.", "Adversarial"),
    ("Chiến lược (strategy) trong trò chơi là gì?", "Chiến lược chỉ định hành động cho mọi phản hồi có thể của đối thủ, không chỉ một chuỗi cố định.", "Adversarial"),
    ("Trò chơi thông tin hoàn hảo là gì?", "Người chơi quan sát đầy đủ trạng thái và lịch sử nước đi (ví dụ cờ vua).", "Adversarial"),
    ("Trò chơi zero-sum là gì?", "Tổng lợi ích bằng 0: một bên được bao nhiêu bên kia mất bấy nhiêu.", "Adversarial"),

    # ===== CSP =====
    ("Bài toán CSP là gì?", "CSP là gán giá trị cho biến sao cho thỏa tất cả ràng buộc; mỗi biến có miền giá trị.", "CSP"),
    ("Một CSP gồm những thành phần nào?", "Gồm: tập biến, miền giá trị cho mỗi biến, và tập ràng buộc.", "CSP"),
    ("Ràng buộc đơn biến là gì?", "Ràng buộc áp dụng lên một biến (ví dụ biến không được nhận giá trị nào đó).", "CSP"),
    ("Ràng buộc nhị phân là gì?", "Ràng buộc liên hệ giữa hai biến (ví dụ hai biến không được cùng màu).", "CSP"),
    ("Đồ thị ràng buộc là gì?", "Biểu diễn CSP: nút là biến, cạnh là ràng buộc giữa hai biến.", "CSP"),
    ("Backtracking trong CSP là gì?", "DFS gán từng biến; bế tắc thì quay lui thử giá trị khác.", "CSP"),
    ("MRV là gì?", "Chọn biến có ít giá trị hợp lệ còn lại nhất để gán trước.", "CSP"),
    ("Degree heuristic là gì?", "Ưu tiên biến ràng buộc nhiều biến chưa gán nhất để giảm nhánh.", "CSP"),
    ("LCV là gì?", "Chọn giá trị ít hạn chế lựa chọn của biến khác nhất.", "CSP"),
    ("Forward checking là gì?", "Loại bỏ sớm giá trị không hợp lệ khỏi miền của biến chưa gán sau khi gán một biến.", "CSP"),
    ("Arc consistency là gì?", "Mỗi giá trị của X phải có ít nhất một giá trị của Y thỏa ràng buộc (đối với cung X→Y).", "CSP"),

    # ===== LOGIC =====
    ("Logic mệnh đề là gì?", "Logic mệnh đề dùng các mệnh đề đúng/sai và phép nối để biểu diễn và suy luận.", "Logic"),
    ("Logic vị từ bậc nhất là gì?", "FOL mở rộng logic mệnh đề bằng đối tượng, biến, hàm, vị từ và lượng từ.", "Logic"),
    ("Cú pháp (syntax) trong logic là gì?", "Syntax là quy tắc tạo công thức hợp lệ trong ngôn ngữ logic.", "Logic"),
    ("Ngữ nghĩa (semantics) trong logic là gì?", "Semantics là ý nghĩa/diễn giải của ký hiệu và điều kiện để mệnh đề đúng/sai.", "Logic"),
    ("Luật De Morgan là gì?", "¬(A∧B)≡(¬A∨¬B) và ¬(A∨B)≡(¬A∧¬B).", "Logic"),
    ("CNF là gì?", "CNF là AND của các clause; mỗi clause là OR của các literal.", "Logic"),
    ("DNF là gì?", "DNF là OR của các mệnh đề; mỗi mệnh đề là AND của các literal.", "Logic"),
    ("SAT là gì?", "SAT là bài toán kiểm tra một công thức có phép gán làm nó đúng hay không.", "Logic"),
    ("Hợp giải (resolution) là gì?", "Luật suy diễn trên CNF để suy ra clause mới, thường dùng trong chứng minh phản chứng.", "Logic"),
    ("Chứng minh phản chứng là gì?", "Giả sử phủ định kết luận rồi suy ra mâu thuẫn để kết luận mệnh đề ban đầu đúng.", "Logic"),
    ("Lượng từ ∀ nghĩa là gì?", "∀x φ(x): với mọi x, φ(x) đúng.", "Logic"),
    ("Lượng từ ∃ nghĩa là gì?", "∃x φ(x): tồn tại ít nhất một x sao cho φ(x) đúng.", "Logic"),
    ("Phủ định lượng từ đổi thế nào?", "¬∀xφ ≡ ∃x¬φ và ¬∃xφ ≡ ∀x¬φ.", "Logic"),
    ("Hợp nhất (unification) là gì?", "Tìm phép thế biến để hai biểu thức trở nên giống nhau.", "Logic"),
    ("Suy diễn tiến là gì?", "Suy diễn tiến suy ra факт mới từ dữ liệu và luật cho đến khi đạt mục tiêu.", "Logic"),
    ("Suy diễn lùi là gì?", "Suy diễn lùi bắt đầu từ mục tiêu rồi truy ngược luật để tìm điều kiện cần.", "Logic"),

    # ===== KNOWLEDGE =====
    ("Biểu diễn tri thức là gì?", "Là cách mô tả tri thức để máy có thể lưu trữ và suy luận.", "Knowledge"),
    ("Knowledge Base là gì?", "Là tập câu (facts/rules) trong ngôn ngữ hình thức mô tả thế giới.", "Knowledge"),
    ("Inference engine là gì?", "Là thủ tục suy luận để rút ra tri thức mới hoặc quyết định hành động từ KB.", "Knowledge"),
    ("Frame là gì?", "Frame biểu diễn đối tượng theo slot (thuộc tính) và giá trị; hỗ trợ thừa kế.", "Knowledge"),
    ("Ontology là gì?", "Ontology là đặc tả khái niệm và quan hệ trong một miền tri thức.", "Knowledge"),
    ("Taxonomy là gì?", "Taxonomy là phân cấp khái niệm (is-a) hỗ trợ thừa kế thuộc tính.", "Knowledge"),
    ("Thừa kế thuộc tính là gì?", "Khái niệm con có thể kế thừa thuộc tính từ khái niệm cha.", "Knowledge"),
    ("Reification là gì?", "Biến một quan hệ/thuộc tính thành đối tượng để có thể nói về nó trong KB.", "Knowledge"),

    # ===== MACHINE LEARNING =====
    ("Học máy (Machine Learning) là gì?", "Học máy là phương pháp giúp hệ thống cải thiện hiệu năng nhờ dữ liệu/kinh nghiệm.", "MachineLearning"),
    ("Học có giám sát là gì?", "Học từ dữ liệu có nhãn để dự đoán nhãn/giá trị mới.", "MachineLearning"),
    ("Học không giám sát là gì?", "Học từ dữ liệu không nhãn để tìm cấu trúc ẩn (cụm, biểu diễn).", "MachineLearning"),
    ("Học tăng cường là gì?", "Học qua tương tác với môi trường để tối đa hóa phần thưởng tích lũy.", "MachineLearning"),
    ("Phân loại là gì?", "Dự đoán nhãn rời rạc cho dữ liệu.", "MachineLearning"),
    ("Hồi quy là gì?", "Dự đoán giá trị liên tục.", "MachineLearning"),
    ("Đặc trưng (feature) là gì?", "Thuộc tính đầu vào dùng để mô hình học và dự đoán.", "MachineLearning"),
    ("Nhãn (label) là gì?", "Đầu ra mục tiêu gắn với dữ liệu trong học có giám sát.", "MachineLearning"),
    ("Overfitting là gì?", "Mô hình khớp quá sát train nên kém trên dữ liệu mới.", "MachineLearning"),
    ("Underfitting là gì?", "Mô hình quá đơn giản nên không học được quy luật.", "MachineLearning"),
    ("Cross-validation dùng để làm gì?", "Đánh giá tổng quát ổn định bằng chia dữ liệu thành nhiều fold huấn luyện/đánh giá luân phiên.", "MachineLearning"),
    ("Siêu tham số là gì?", "Tham số do người thiết kế chọn trước huấn luyện (ví dụ k của KNN).", "MachineLearning"),
    ("Regularization là gì?", "Thêm ràng buộc/phạt để giảm overfitting.", "MachineLearning"),
    ("Confusion matrix là gì?", "Bảng tổng hợp dự đoán đúng/sai theo lớp (TP/FP/TN/FN).", "MachineLearning"),
    ("Accuracy là gì?", "Tỷ lệ dự đoán đúng trên tổng số mẫu.", "MachineLearning"),
    ("Precision là gì?", "TP/(TP+FP): đúng trong các dự đoán dương.", "MachineLearning"),
    ("Recall là gì?", "TP/(TP+FN): tìm được bao nhiêu mẫu dương thực.", "MachineLearning"),
    ("F1-score là gì?", "Trung bình điều hòa của precision và recall.", "MachineLearning"),
    ("TF-IDF là gì?", "Biểu diễn văn bản theo TF (tần suất trong tài liệu) và IDF (độ hiếm trong tập).", "MachineLearning"),
    ("Bag-of-Words là gì?", "Biểu diễn văn bản bằng vector đếm/tần suất từ, bỏ qua thứ tự từ.", "MachineLearning"),
    ("Embedding là gì?", "Biểu diễn vector liên tục sao cho gần nhau phản ánh tương đồng ngữ nghĩa.", "MachineLearning"),
    ("KNN là gì?", "KNN dự đoán dựa trên k láng giềng gần nhất theo một độ đo khoảng cách.", "MachineLearning"),
    ("Vì sao cần chuẩn hóa dữ liệu cho KNN?", "Chuẩn hóa tránh đặc trưng có thang lớn lấn át khoảng cách, giúp KNN công bằng hơn.", "MachineLearning"),
    ("Naive Bayes là gì?", "Phân loại xác suất dựa trên Bayes với giả định độc lập có điều kiện giữa các đặc trưng.", "MachineLearning"),
    ("Xác suất tiên nghiệm (prior) là gì?", "P(c): xác suất của lớp/giả thiết trước khi quan sát dữ liệu.", "MachineLearning"),
    ("Likelihood là gì?", "P(x|c): xác suất quan sát x nếu lớp c đúng.", "MachineLearning"),
    ("Xác suất hậu nghiệm (posterior) là gì?", "P(c|x): xác suất lớp c đúng sau khi quan sát x.", "MachineLearning"),
    ("MAP là gì?", "Chọn lớp có P(c|x) lớn nhất, tương đương tối đa hóa P(x|c)P(c).", "MachineLearning"),
    ("Laplace smoothing là gì?", "Cộng hằng nhỏ vào tần suất để tránh xác suất 0 trong Naive Bayes.", "MachineLearning"),
    ("Calibration là gì?", "Độ hiệu chuẩn: xác suất dự đoán phản ánh đúng tần suất thực tế.", "MachineLearning"),
    ("Temperature Scaling là gì?", "Kỹ thuật hiệu chuẩn xác suất bằng cách chia logit cho nhiệt độ T học trên validation.", "MachineLearning"),
    ("Sigmoid (Platt) scaling là gì?", "Hiệu chuẩn bằng cách học hàm sigmoid ánh xạ điểm số mô hình sang xác suất.", "MachineLearning"),
]

# Bổ sung thêm khái niệm ML (lý thuyết ngoài) để đủ đa dạng nhưng vẫn ngắn
EXTRA_GLOSSARY = [
    ("Logistic Regression", "Mô hình phân loại tuyến tính ước lượng xác suất lớp bằng sigmoid/softmax.", "MachineLearning"),
    ("SVM", "Mô hình tìm siêu phẳng biên lớn nhất; có thể dùng kernel cho phi tuyến.", "MachineLearning"),
    ("Decision Tree", "Cây quyết định chia theo điều kiện đặc trưng; lá là dự đoán.", "MachineLearning"),
    ("Random Forest", "Tập hợp nhiều cây quyết định huấn luyện ngẫu nhiên; giảm overfitting nhờ ensemble.", "MachineLearning"),
    ("Gradient Descent", "Thuật toán tối ưu cập nhật tham số theo hướng ngược gradient để giảm loss.", "MachineLearning"),
    ("Early stopping", "Dừng huấn luyện khi validation không cải thiện để giảm overfitting.", "MachineLearning"),
    ("Dropout", "Regularization cho mạng nơ-ron bằng cách ngẫu nhiên bỏ neuron khi huấn luyện.", "MachineLearning"),
    ("Bias-Variance tradeoff", "Cân bằng giữa sai số do đơn giản (bias) và do nhạy dữ liệu (variance).", "MachineLearning"),
    ("Data leakage", "Rò rỉ thông tin từ test/validation vào huấn luyện làm đánh giá sai.", "MachineLearning"),
]

# ----------------------------
# 2) PARAPHRASE (đảo câu)
# ----------------------------
LA_GI = [
    "{x} là gì?", "Khái niệm {x} là gì?", "Định nghĩa {x}.", "Bạn hãy định nghĩa {x}.",
    "{x} nghĩa là gì?", "{x} được hiểu là gì?", "Giải thích {x}.", "Cho biết {x} là gì.",
    "Nói ngắn gọn {x} là gì?", "{x} là thuật ngữ gì?", "Bạn hiểu {x} là gì?",
]
HOW = [
    "{x} hoạt động như thế nào?", "Cơ chế hoạt động của {x} là gì?", "{x} vận hành ra sao?",
    "Nguyên lý hoạt động của {x}?", "{x} hoạt động theo cơ chế nào?", "Mô tả cách {x} hoạt động.",
]
PURPOSE = [
    "{x} dùng để làm gì?", "Mục đích của {x} là gì?", "Vai trò của {x} là gì?",
    "{x} có tác dụng gì?", "Khi nào dùng {x}?", "Ứng dụng của {x} là gì?",
]
COMPONENTS = [
    "{x} gồm những gì?", "{x} bao gồm những thành phần nào?", "Các thành phần của {x} là gì?",
    "{x} có những phần nào?", "Hãy nêu các thành phần của {x}.",
]
LEADINS = ["", "Trong AI, ", "Trong học máy, ", "Theo lý thuyết, ", "Cho mình hỏi ", "Bạn có thể cho biết "]
TAILS = ["", "", "", " (nêu ý chính)", " (giải thích ngắn)"]

def norm_q(q: str) -> str:
    q = re.sub(r"\s+", " ", q.strip())
    if not q.endswith("?") and not q.endswith("."):
        q += "?"
    return q

def detect_and_paraphrase(q: str) -> List[str]:
    q0 = q.strip().replace("？", "?")
    # X là gì?
    m = re.match(r"^(.*)\s+là\s+gì\?\s*$", q0, flags=re.IGNORECASE)
    if m:
        x = m.group(1).strip()
        out = [t.format(x=x) for t in LA_GI]
        # nếu là từ viết tắt ngắn, thêm biến thể "viết tắt"
        if (len(x) <= 6 and x.upper() == x) or ("(" in x and ")" in x):
            out.append(f"{x} viết tắt của gì?")
            out.append(f"{x} là viết tắt của cụm từ nào?")
        return [norm_q(s) for s in out]

    # X hoạt động như thế nào?
    m = re.match(r"^(.*)\s+hoạt\s+động\s+như\s+thế\s+nào\?\s*$", q0, flags=re.IGNORECASE)
    if m:
        x = m.group(1).strip()
        return [norm_q(t.format(x=x)) for t in HOW]

    # X dùng để làm gì?
    m = re.match(r"^(.*)\s+dùng\s+để\s+làm\s+gì\?\s*$", q0, flags=re.IGNORECASE)
    if m:
        x = m.group(1).strip()
        return [norm_q(t.format(x=x)) for t in PURPOSE]

    # X gồm những ... nào?
    m = re.match(r"^(.*)\s+gồm\s+những\s+thành\s+phần\s+nào\?\s*$", q0, flags=re.IGNORECASE)
    if m:
        x = m.group(1).strip()
        return [norm_q(t.format(x=x)) for t in COMPONENTS]

    # generic: thêm vài biến thể nhẹ
    core = q0[:-1] if q0.endswith("?") else q0
    out = [norm_q(q0)]
    out += [norm_q(f"{p}{core}") for p in ["Giải thích ", "Nêu ngắn gọn ", "Bạn hãy giải thích "]]
    return list(dict.fromkeys(out))

def expand_items(items: List[Tuple[str, str, str]], target_n: int, max_per_base: int = 40) -> List[Tuple[str, str, str]]:
    seen = set()
    out: List[Tuple[str, str, str]] = []

    for q, a, topic in items:
        variants = detect_and_paraphrase(q)

        # thêm prefix/suffix để đảo ngôn từ mạnh nhưng vẫn ngắn
        enriched = []
        for v in variants:
            core = v[:-1] if v.endswith("?") else v
            for pre in LEADINS:
                for tail in TAILS:
                    if pre or tail:
                        enriched.append(norm_q(f"{pre}{core}{tail}"))
        variants = list(dict.fromkeys(variants + enriched))
        random.shuffle(variants)
        variants = variants[:max_per_base]

        for v in variants:
            key = v.lower()
            if key not in seen:
                out.append((v, a, topic))
                seen.add(key)

    # nếu chưa đủ, bổ sung glossary rồi mở rộng tiếp
    if len(out) < target_n:
        gloss = [(f"{x} là gì?", ans, topic) for (x, ans, topic) in EXTRA_GLOSSARY]
        for q, a, topic in gloss:
            for v in detect_and_paraphrase(q):
                key = v.lower()
                if key not in seen:
                    out.append((v, a, topic))
                    seen.add(key)
                if len(out) >= target_n:
                    break
            if len(out) >= target_n:
                break

    # vẫn thiếu thì nhân thêm lead-in ngắn (không làm câu dài quá)
    if len(out) < target_n:
        add_lead = ["Cho mình hỏi ", "Bạn cho biết ", "Giải thích giúp mình "]
        extra = []
        for q, a, topic in out:
            core = q[:-1] if q.endswith("?") else q
            for li in add_lead:
                extra.append((norm_q(li + core), a, topic))
        for q, a, topic in extra:
            key = q.lower()
            if key not in seen:
                out.append((q, a, topic))
                seen.add(key)
            if len(out) >= target_n:
                break

    return out[:target_n]

def sql_escape(s: str) -> str:
    return s.replace("'", "''")

def write_sql(rows: List[Tuple[str, str, str]], outpath: str, chunk_size: int = 300) -> None:
    header = """-- ===================================================
-- init.sql – Khởi tạo cơ sở dữ liệu chatbot AI
-- ===================================================

CREATE TABLE IF NOT EXISTS qa (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    question TEXT NOT NULL,
    answer TEXT NOT NULL,
    topic TEXT NOT NULL
);

"""
    with open(outpath, "w", encoding="utf-8") as f:
        f.write(header)

        # chunked INSERT for readability
        for i in range(0, len(rows), chunk_size):
            chunk = rows[i:i+chunk_size]
            f.write("INSERT INTO qa (question, answer, topic) VALUES\n")
            vals = []
            for q, a, topic in chunk:
                vals.append(f"('{sql_escape(q)}', '{sql_escape(a)}', '{sql_escape(topic)}')")
            f.write(",\n".join(vals))
            f.write(";\n\n")

def write_csv(rows: List[Tuple[str, str, str]], outpath: str) -> None:
    with open(outpath, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["question", "answer", "topic"])
        for q, a, topic in rows:
            w.writerow([q, a, topic])

def write_jsonl(rows: List[Tuple[str, str, str]], outpath: str) -> None:
    with open(outpath, "w", encoding="utf-8") as f:
        for q, a, topic in rows:
            f.write(json.dumps({"question": q, "answer": a, "topic": topic}, ensure_ascii=False) + "\n")

def write_sqlite(rows: List[Tuple[str, str, str]], outpath: str) -> None:
    if os.path.exists(outpath):
        os.remove(outpath)
    conn = sqlite3.connect(outpath)
    cur = conn.cursor()
    cur.execute("""
    CREATE TABLE IF NOT EXISTS qa (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        question TEXT NOT NULL,
        answer TEXT NOT NULL,
        topic TEXT NOT NULL
    );
    """)
    cur.executemany("INSERT INTO qa (question, answer, topic) VALUES (?, ?, ?);", rows)
    conn.commit()
    conn.close()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=2000, help="number of Q&A rows")
    ap.add_argument("--outdir", type=str, default=".", help="output directory")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    rows = expand_items(BASE_ITEMS, target_n=args.n, max_per_base=40)

    write_sql(rows, os.path.join(args.outdir, "init.sql"))
    write_csv(rows, os.path.join(args.outdir, "qa_2000.csv"))
    write_jsonl(rows, os.path.join(args.outdir, "qa_2000.jsonl"))
    write_sqlite(rows, os.path.join(args.outdir, "chatbot_ai.db"))

    # quick stats
    topics = {}
    for _, _, t in rows:
        topics[t] = topics.get(t, 0) + 1
    print(f"Generated {len(rows)} rows")
    print("Topic counts:", topics)

if __name__ == "__main__":
    main()
