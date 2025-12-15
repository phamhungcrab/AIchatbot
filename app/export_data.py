import pandas as pd
import sqlite3
import os

# Define paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "..", "data", "knowledge.db")
OUTPUT_PATH = os.path.join(BASE_DIR, "..", "data", "train_data_full.csv")

def export_data():
    print(f"📂 Connecting to database at {DB_PATH}...")
    
    if not os.path.exists(DB_PATH):
        print("❌ Database not found! Please run app/datastore.py first to initialize it.")
        return

    try:
        with sqlite3.connect(DB_PATH) as conn:
            # Select all columns including topic
            df = pd.read_sql_query("SELECT question, answer, topic FROM qa", conn)
            
        print(f"✅ Fetched {len(df)} rows.")
        
        # Save to CSV
        df.to_csv(OUTPUT_PATH, index=False, encoding='utf-8')
        print(f"💾 Data exported to {OUTPUT_PATH}")
        print("👉 Please upload THIS file ('train_data_full.csv') to Google Colab for training.")
        
    except Exception as e:
        print(f"❌ Error exporting data: {e}")

if __name__ == "__main__":
    export_data()
