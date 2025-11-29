import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Trainer, TrainingArguments
import os

# 1. Cấu hình
MODEL_NAME = "google/mt5-small" # Model đa ngôn ngữ, hỗ trợ tốt tiếng Việt
MAX_LEN = 128
EPOCHS = 5 # Số vòng lặp huấn luyện
BATCH_SIZE = 4

# Đường dẫn
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, '..', 'data', 'train_data.csv')
OUTPUT_DIR = os.path.join(BASE_DIR, '..', 'models', 'my_generative_bot')

# 2. Chuẩn bị Dataset
class ChatbotDataset(Dataset):
    def __init__(self, tokenizer, data_path, max_len=128):
        self.data = pd.read_csv(data_path)
        self.tokenizer = tokenizer
        self.max_len = max_len
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        row = self.data.iloc[index]
        question = str(row['question'])
        answer = str(row['answer'])
        
        # T5 yêu cầu format: "translate English to German: ..." (ví dụ)
        # Ở đây ta dùng prefix đơn giản hoặc không cần
        input_text = f"question: {question}"
        target_text = f"{answer}"
        
        # Tokenize Input
        input_encoding = self.tokenizer(
            input_text,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        # Tokenize Target
        target_encoding = self.tokenizer(
            target_text,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        labels = target_encoding.input_ids
        labels[labels == self.tokenizer.pad_token_id] = -100 # Bỏ qua padding khi tính loss
        
        return {
            "input_ids": input_encoding.input_ids.flatten(),
            "attention_mask": input_encoding.attention_mask.flatten(),
            "labels": labels.flatten()
        }

def train():
    print("🚀 Đang chuẩn bị huấn luyện Generative Bot...")
    
    # Check GPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"⚙️ Thiết bị sử dụng: {device.upper()}")
    
    # Load Tokenizer & Model
    print(f"📥 Đang tải model {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
    model.to(device)
    
    # Load Data
    print(f"📚 Đang đọc dữ liệu từ {DATA_PATH}...")
    dataset = ChatbotDataset(tokenizer, DATA_PATH, MAX_LEN)
    
    # Training Arguments
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        save_steps=500,
        save_total_limit=2,
        logging_steps=10,
        learning_rate=3e-4,
        remove_unused_columns=False # Quan trọng với custom dataset
    )
    
    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
    )
    
    # Start Training
    print("🏋️‍♂️ Bắt đầu huấn luyện (Fine-tuning)...")
    trainer.train()
    
    # Save Model
    print("💾 Đang lưu model...")
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    
    print(f"✅ Hoàn tất! Model đã lưu tại: {OUTPUT_DIR}")

if __name__ == "__main__":
    train()
