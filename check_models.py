import os
import google.generativeai as genai
from dotenv import load_dotenv

# Load .env
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ENV_PATH = os.path.join(BASE_DIR, '.env')
load_dotenv(dotenv_path=ENV_PATH)

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    print("❌ Không tìm thấy API Key in .env")
else:
    print(f"✅ Found API Key: {api_key[:5]}...")
    genai.configure(api_key=api_key)
    
    print("\n🔍 Đang lấy danh sách models...")
    try:
        for m in genai.list_models():
            if 'generateContent' in m.supported_generation_methods:
                print(f"- {m.name}")
    except Exception as e:
        print(f"❌ Lỗi: {e}")
