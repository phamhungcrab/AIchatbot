-- ===================================================
-- 📘 init.sql – Khởi tạo cơ sở dữ liệu chatbot AI
-- ===================================================

CREATE TABLE IF NOT EXISTS qa (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    question TEXT NOT NULL,
    answer TEXT NOT NULL,
    topic TEXT NOT NULL
);

-- ===================================================
-- 🔹 Nạp dữ liệu Q&A (tổng cộng 78 câu, chia theo topic)
-- ===================================================

INSERT INTO qa (question, answer, topic) VALUES

-- 🧠 MACHINE LEARNING (15 câu)
('KNN là gì?', 'K-Nearest Neighbors (KNN) là thuật toán học máy thuộc nhóm giám sát, phân loại dữ liệu mới dựa trên khoảng cách đến các điểm lân cận trong tập huấn luyện.', 'MachineLearning'),
('Naïve Bayes hoạt động như thế nào?', 'Naïve Bayes dựa trên định lý Bayes, giả định các đặc trưng độc lập có điều kiện, tính xác suất của mỗi lớp và chọn lớp có xác suất cao nhất.', 'MachineLearning'),
('Học có giám sát là gì?', 'Là quá trình học từ dữ liệu có nhãn sẵn, mô hình học mối quan hệ giữa đầu vào và đầu ra để dự đoán nhãn mới.', 'MachineLearning'),
('Học không giám sát là gì?', 'Là quá trình học từ dữ liệu không có nhãn, mô hình tìm ra cấu trúc ẩn như cụm hoặc quan hệ giữa các điểm dữ liệu.', 'MachineLearning'),
('Học tăng cường là gì?', 'Là dạng học trong đó tác tử học thông qua tương tác với môi trường để tối đa hóa phần thưởng tích lũy.', 'MachineLearning'),
('Overfitting là gì?', 'Là hiện tượng mô hình học quá sát dữ liệu huấn luyện, dẫn đến hoạt động kém trên dữ liệu mới.', 'MachineLearning'),
('Underfitting là gì?', 'Là khi mô hình quá đơn giản, không học được quy luật trong dữ liệu, nên kết quả huấn luyện và dự đoán đều kém.', 'MachineLearning'),
('Đặc trưng (feature) trong học máy là gì?', 'Là các thuộc tính đầu vào mô tả dữ liệu dùng để mô hình học ra mối quan hệ với đầu ra.', 'MachineLearning'),
('Phân loại (classification) là gì?', 'Là nhiệm vụ dự đoán nhãn lớp rời rạc cho dữ liệu dựa trên tập huấn luyện.', 'MachineLearning'),
('Hồi quy (regression) là gì?', 'Là nhiệm vụ dự đoán giá trị liên tục, ví dụ như giá nhà, nhiệt độ...', 'MachineLearning'),
('Cross-validation dùng để làm gì?', 'Dùng để đánh giá độ tổng quát của mô hình bằng cách chia dữ liệu thành nhiều phần để huấn luyện và kiểm thử.', 'MachineLearning'),
('Tập huấn luyện và tập kiểm thử khác nhau thế nào?', 'Tập huấn luyện dùng để học mô hình, tập kiểm thử dùng để đánh giá độ chính xác của mô hình trên dữ liệu chưa thấy.', 'MachineLearning'),
('Định lý Bayes nói gì?', 'P(H|D) = P(D|H) * P(H) / P(D), cho phép cập nhật xác suất giả thuyết H dựa trên dữ liệu quan sát D.', 'MachineLearning'),
('TF-IDF là gì?', 'TF-IDF là phương pháp biểu diễn văn bản dựa trên tần suất từ (TF) và tầm quan trọng của từ trong toàn bộ tập dữ liệu (IDF).', 'MachineLearning'),
('Thuật toán K-means là gì?', 'Là thuật toán học không giám sát để phân cụm dữ liệu bằng cách tối thiểu hóa khoảng cách trong từng cụm.', 'MachineLearning'),

-- 🤖 AGENTS (10 câu)
('Tác tử là gì?', 'Tác tử (Agent) là thực thể có khả năng cảm nhận môi trường xung quanh qua cảm biến và hành động thông qua bộ phận chấp hành.', 'Agents'),
('PEAS gồm những thành phần nào?', 'PEAS gồm bốn thành phần: Performance measure, Environment, Actuators, Sensors – dùng để mô tả tác tử và môi trường hoạt động.', 'Agents'),
('Tác tử hợp lý là gì?', 'Là tác tử luôn chọn hành động giúp tối đa hóa tiêu chí đánh giá hiệu quả hoạt động dựa trên nhận thức hiện có.', 'Agents'),
('Tác tử phản xạ đơn giản là gì?', 'Là tác tử chọn hành động chỉ dựa trên trạng thái hiện tại, không ghi nhớ lịch sử.', 'Agents'),
('Tác tử dựa trên mô hình là gì?', 'Là tác tử có mô hình về cách thế giới hoạt động để dự đoán trạng thái tiếp theo.', 'Agents'),
('Tác tử mục tiêu là gì?', 'Là tác tử lựa chọn hành động để đạt được một mục tiêu nhất định thay vì phản ứng đơn thuần.', 'Agents'),
('Tác tử tiện ích là gì?', 'Là tác tử đánh giá hành động dựa trên hàm tiện ích, cho phép so sánh mức độ mong muốn của các trạng thái.', 'Agents'),
('Tác tử tự trị là gì?', 'Là tác tử có thể ra quyết định dựa trên kinh nghiệm và tri thức của chính nó mà không cần can thiệp bên ngoài.', 'Agents'),
('Môi trường có thể quan sát hoàn toàn là gì?', 'Là môi trường trong đó tác tử biết đầy đủ trạng thái hiện tại của nó.', 'Agents'),
('Môi trường ngẫu nhiên là gì?', 'Là môi trường mà kết quả hành động của tác tử không thể dự đoán chính xác do yếu tố ngẫu nhiên.', 'Agents'),

-- 🔍 SEARCH (12 câu)
('Tìm kiếm theo chiều rộng (BFS) là gì?', 'BFS mở rộng các nút ở cùng độ sâu trước khi đi sâu hơn – thường dùng hàng đợi (queue) để lưu trữ trạng thái.', 'Search'),
('DFS khác BFS như thế nào?', 'DFS đi sâu theo nhánh đầu tiên, dùng ngăn xếp; BFS đi theo từng lớp, dùng hàng đợi.', 'Search'),
('Tìm kiếm theo chiều sâu lặp lại là gì?', 'Là sự kết hợp giữa DFS và BFS, lặp dần theo giới hạn độ sâu để đảm bảo tính hoàn chỉnh.', 'Search'),
('Chi phí đường đi là gì?', 'Là tổng chi phí của các hành động từ trạng thái gốc đến trạng thái hiện tại.', 'Search'),
('Thuật toán A* là gì?', 'A* là thuật toán tìm kiếm tối ưu sử dụng hàm f(n) = g(n) + h(n), trong đó g là chi phí đã đi và h là ước lượng còn lại.', 'Search'),
('Hàm heuristic là gì?', 'Là hàm ước lượng chi phí còn lại từ trạng thái hiện tại đến trạng thái đích, giúp tìm kiếm hiệu quả hơn.', 'Search'),
('Best First Search là gì?', 'Là thuật toán chọn mở rộng nút có giá trị heuristic nhỏ nhất, hướng đến đích nhanh.', 'Search'),
('Uniform Cost Search là gì?', 'Là thuật toán mở rộng nút có chi phí đường đi nhỏ nhất, đảm bảo tìm được đường đi tối ưu.', 'Search'),
('Hill Climbing là gì?', 'Là thuật toán leo đồi, luôn di chuyển đến trạng thái có giá trị tốt hơn, dễ mắc kẹt tại cực trị địa phương.', 'Search'),
('Simulated Annealing là gì?', 'Là thuật toán tìm kiếm ngẫu nhiên, cho phép chấp nhận bước tồi tạm thời để tránh mắc kẹt cực trị địa phương.', 'Search'),
('Genetic Algorithm là gì?', 'Là thuật toán mô phỏng tiến hóa sinh học qua chọn lọc, lai ghép và đột biến để tối ưu nghiệm.', 'Search'),
('Beam Search là gì?', 'Là thuật toán tìm kiếm song song, chỉ giữ lại k trạng thái tốt nhất ở mỗi bước mở rộng.', 'Search'),

-- 🔢 LOGIC (12 câu)
('Logic mệnh đề là gì?', 'Logic mệnh đề là hệ thống logic trong đó các phát biểu có giá trị Đúng hoặc Sai.', 'Logic'),
('Logic vị từ khác logic mệnh đề ở điểm nào?', 'Logic vị từ mở rộng logic mệnh đề bằng biến, hàm và lượng từ.', 'Logic'),
('Lượng từ tồn tại nghĩa là gì?', 'Là ký hiệu ∃x, biểu thị “tồn tại ít nhất một x” thoả mãn mệnh đề.', 'Logic'),
('Lượng từ với mọi nghĩa là gì?', 'Là ký hiệu ∀x, biểu thị “với mọi x” mệnh đề đều đúng.', 'Logic'),
('Mệnh đề kéo theo là gì?', 'Là mệnh đề có dạng P → Q, chỉ sai khi P đúng và Q sai.', 'Logic'),
('Phủ định của một mệnh đề là gì?', 'Là mệnh đề có giá trị chân lý ngược lại so với mệnh đề ban đầu.', 'Logic'),
('Luật De Morgan là gì?', 'Là quy tắc: ¬(A ∧ B) = ¬A ∨ ¬B và ¬(A ∨ B) = ¬A ∧ ¬B.', 'Logic'),
('Hợp nhất trong logic vị từ là gì?', 'Là quá trình tìm phép gán biến để hai biểu thức logic trở nên giống nhau.', 'Logic'),
('Suy diễn tiến là gì?', 'Là quá trình áp dụng luật từ dữ liệu hiện có để suy ra tri thức mới.', 'Logic'),
('Suy diễn lùi là gì?', 'Là quá trình bắt đầu từ mục tiêu rồi truy ngược lại các điều kiện cần thỏa.', 'Logic'),
('Chứng minh bằng phản chứng là gì?', 'Là chứng minh mệnh đề P đúng bằng cách giả sử P sai và dẫn đến mâu thuẫn.', 'Logic'),
('CNF (Conjunctive Normal Form) là gì?', 'Là dạng chuẩn của biểu thức logic, biểu diễn bằng tích của các tổng.', 'Logic'),

-- 🧩 KNOWLEDGE REPRESENTATION (10 câu)
('Biểu diễn tri thức là gì?', 'Là cách thức mô tả tri thức trong máy tính để hệ thống có thể suy luận được.', 'Knowledge'),
('Luật sản xuất là gì?', 'Là dạng IF-THEN biểu diễn mối quan hệ giữa điều kiện và hành động.', 'Knowledge'),
('Mạng ngữ nghĩa là gì?', 'Là đồ thị gồm các đỉnh biểu diễn khái niệm và cạnh biểu diễn quan hệ giữa chúng.', 'Knowledge'),
('Khung (Frame) là gì?', 'Là cấu trúc dữ liệu mô tả đối tượng bằng các thuộc tính và giá trị.', 'Knowledge'),
('Ontology là gì?', 'Là tập hợp các khái niệm và quan hệ được định nghĩa trong một miền tri thức.', 'Knowledge'),
('Logic mờ (Fuzzy Logic) là gì?', 'Là logic cho phép giá trị chân lý nằm giữa 0 và 1, thay vì chỉ đúng hoặc sai.', 'Knowledge'),
('Tri thức khai báo là gì?', 'Là tri thức mô tả sự vật, sự việc bằng các phát biểu, không chỉ rõ cách thực hiện.', 'Knowledge'),
('Tri thức thủ tục là gì?', 'Là tri thức mô tả cách thức thực hiện một nhiệm vụ, ví dụ như thuật toán.', 'Knowledge'),
('Hệ chuyên gia là gì?', 'Là hệ thống sử dụng tri thức chuyên môn để giải quyết vấn đề như con người.', 'Knowledge'),
('Inference Engine là gì?', 'Là thành phần của hệ chuyên gia dùng để suy luận từ tri thức có sẵn.', 'Knowledge'),

-- 🎮 AI PROJECTS / ỨNG DỤNG (9 câu)
('Ứng dụng của AI trong y tế là gì?', 'AI giúp chẩn đoán hình ảnh, phân tích gen và hỗ trợ bác sĩ ra quyết định điều trị.', 'Applications'),
('Ứng dụng của AI trong giáo dục là gì?', 'AI có thể cá nhân hóa học tập, chấm bài tự động và hỗ trợ học sinh luyện tập.', 'Applications'),
('Ứng dụng của AI trong giao thông là gì?', 'AI được dùng trong xe tự lái, quản lý luồng giao thông và dự báo tắc đường.', 'Applications'),
('Ứng dụng của AI trong thương mại điện tử là gì?', 'AI phân tích hành vi mua sắm, gợi ý sản phẩm và tối ưu trải nghiệm khách hàng.', 'Applications'),
('Chatbot AI là gì?', 'Là chương trình có khả năng hiểu và phản hồi ngôn ngữ tự nhiên để giao tiếp với người dùng.', 'Applications'),
('AlphaGo là gì?', 'AlphaGo là hệ thống AI của Google DeepMind đã đánh bại kỳ thủ cờ vây hàng đầu thế giới.', 'Applications'),
('GAN là gì?', 'Generative Adversarial Network gồm hai mạng đối kháng để sinh ra dữ liệu mới giống thật.', 'Applications'),
('Robot thông minh khác robot thường ở điểm nào?', 'Robot thông minh có khả năng học, nhận thức và thích nghi với môi trường.', 'Applications'),
('Tương lai của AI là gì?', 'AI sẽ phát triển mạnh trong tự động hóa, sáng tạo nội dung, y sinh học và giáo dục thông minh.', 'Applications');
