#file này test code python
import os
from app.datastore import get_all_qa
from app.preprocess import preprocess_text, train_vectorizer
from app.nb_module import train_naive_bayes
from app.knn_module import train_knn
import pandas as pd
import pickle

print("Thư mục làm việc hiện tại (CWD):")
print(os.getcwd())

df = get_all_qa()
print(df)

df['clean_text'] = df['question'].apply(preprocess_text)
vectorizer = train_vectorizer(df['clean_text'])
print("-" * 10)
print(df)
print("-" * 10)
print(vectorizer)