import pandas as pd
import time
import os
import sys

# Add the parent directory to sys.path to allow imports from app
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.datastore import get_all_qa
from app.genai_module import model

def generate_variations(question, num_variations=25):
    """
    Sử dụng Gemini để sinh ra các biến thể của câu hỏi.
    """
    if not model:
        print("❌ Gemini model chưa được khởi tạo. Kiểm tra API Key.")
        return []

    prompt = f"""
    Hãy viết lại câu hỏi sau đây bằng tiếng Việt theo {num_variations} cách khác nhau nhưng vẫn giữ nguyên ý nghĩa.
    Chỉ liệt kê các câu hỏi, mỗi câu một dòng, không đánh số, không thêm ký tự thừa.
    
    Câu hỏi gốc: "{question}"
    """
    
    retries = 3
    for attempt in range(retries):
        try:
            response = model.generate_content(prompt)
            # Tách dòng và làm sạch
            variations = [line.strip() for line in response.text.split('\n') if line.strip()]
            return variations
        except Exception as e:
            if "429" in str(e) or "Quota exceeded" in str(e):
                wait_time = 60
                print(f"⚠️ Quota exceeded. Waiting {wait_time}s before retrying ({attempt+1}/{retries})...")
                time.sleep(wait_time)
            else:
                print(f"⚠️ Lỗi khi sinh biến thể cho '{question}': {e}")
                return []
    return []

def main():
    print("🚀 Bắt đầu quá trình tăng cường dữ liệu (Data Augmentation)...")
    
    # 1. Lấy dữ liệu gốc
    df = get_all_qa()
    if df.empty:
        print("❌ Không tìm thấy dữ liệu trong database!")
        return

    print(f"📚 Tìm thấy {len(df)} câu hỏi gốc.")
    
    new_data = []
    
    # 2. Duyệt qua từng câu hỏi và sinh biến thể
    # FULL MODE: Xử lý toàn bộ dữ liệu
    print(f"🚀 FULL MODE: Đang xử lý toàn bộ {len(df)} câu hỏi.")

    for index, row in df.iterrows():
        original_q = row['question']
        answer = row['answer']
        
        print(f"[{index+1}/{len(df)}] Đang xử lý: {original_q}")
        
        # Thêm câu gốc trước
        new_data.append({'question': original_q, 'answer': answer})
        
        # Sinh biến thể (chờ 1 chút để không bị rate limit)
        variations = generate_variations(original_q)
        
        for v in variations:
            new_data.append({'question': v, 'answer': answer})
            
        time.sleep(1) # Nghỉ 1 giây giữa các request
        
    # 3. Lưu ra file CSV
    output_df = pd.DataFrame(new_data)
    output_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'train_data.csv')
    output_df.to_csv(output_path, index=False, encoding='utf-8')
    
    print(f"✅ Hoàn tất! Đã tạo ra {len(output_df)} mẫu dữ liệu.")
    print(f"📂 File lưu tại: {output_path}")

if __name__ == "__main__":
    main()
