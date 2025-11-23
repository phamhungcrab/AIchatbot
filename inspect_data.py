import sqlite3
import pandas as pd
import os

# Kết nối DB
db_path = os.path.join(os.path.dirname(__file__), 'data', 'knowledge.db')
conn = sqlite3.connect(db_path)

# Đọc dữ liệu
df = pd.read_sql_query("SELECT * FROM qa", conn)

print(f"Total records: {len(df)}")
print("\nTopic Distribution:")
print(df['topic'].value_counts())

print("\nRecords containing 'KNN' (case-insensitive):")
knn_records = df[df['question'].str.contains('knn', case=False, na=False)]
print(knn_records[['id', 'topic', 'question']])

conn.close()
