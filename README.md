# AIChatbot - Cau truc thu muc trong tam

Muc tieu: liet ke cac thu muc/file lien quan truc tiep den du an chatbot va vai tro cua chung.

## Trong tam (can doc/quan ly khi lam viec voi du an)

- `app/` - Ma nguon chinh cua chatbot (Flask + NLP + suy luan).
  - `chatbot_app.py` - Entrypoint chay web server Flask.
  - `preprocess.py` - Tien xu ly van ban.
  - `nb_module.py` - Phan loai chu de (Naive Bayes).
  - `knn_module.py` - Tim cau hoi gan nhat (KNN/Cosine).
  - `find_answer.py` - Pipeline tim dap an tu du lieu.
  - `response_handler.py` - Xu ly/format phan hoi.
  - `confidence_utils.py` - Tinh do tin cay cho cau tra loi.
  - `create_pkl.py` - Tao vectorizer/model pkl.
  - `simple_evaluation.py` - Danh gia nhanh.
- `data/` - Du lieu train/test dang CSV cho chatbot.
- `models/` - Cac mo hinh da huan luyen (`*.pkl`).
- `templates/` - Giao dien HTML (Jinja2).
- `static/` - Tai nguyen giao dien (CSS/JS/anh).
- `results/` - Bao cao danh gia (Markdown).
- `requirements.txt` - Danh sach thu vien Python can thiet.
- `.env` va `env_example.txt` - Cau hinh moi truong (API key, bien cau hinh).
- `test_accuracy_comparison.py` - Script so sanh do chinh xac.

## Khong trong tam (co the bo qua khi doc code)

- `venv/` - Moi truong ao Python.
- `app/__pycache__/` - Cache tu dong cua Python.
- `temp/` - File tam/thu nghiem.
- `code_lai/` - Ban nhap/ma cu (khong dung trong luong chay chinh).
- `jupyter-report-starters/` - Notebook bao cao.
- `*.png` - Anh minh hoa giao dien.
- `extensions_list.txt` - Danh sach extension moi truong dev.
- `readme.txt` - Ghi chu cu, nen uu tien `README.md`.
- `.DS_Store` - Metadata he dieu hanh.
