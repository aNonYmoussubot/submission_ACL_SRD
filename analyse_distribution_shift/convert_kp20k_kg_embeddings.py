from sentence_transformers import SentenceTransformer
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import numpy as np
import h5py
import json
corpus_path= '../kp20k_clean.json'
from cleantext import clean
from tqdm import tqdm
embed_model_path = 'gte-large'
# 使用 sentence-transformers 加载预训练的模型
model = SentenceTransformer(embed_model_path)

all_clean_texts = []
with open(corpus_path, "r") as fr:
    all_clean_texts = json.load(fr)
print(all_clean_texts[0])

batch_size = 256
num_samples = len(all_clean_texts)
total_batch = num_samples // batch_size + 1
embeddings_dataset = []

output_path = 'embeds/kp20k_train_kp_emb.h5'
with h5py.File(output_path, 'w') as f:
    embeddings_dataset = f.create_dataset('embeddings', (num_samples, 1024), dtype='f4')  # 假设嵌入的维度是 1024
    
    for i in tqdm(range(0, num_samples, batch_size), total=total_batch):
        batch_samples = all_clean_texts[i:i+batch_size]
        batch_samples = [';'.join(item['present_kp'] + item['absent_kp'])for item in batch_samples]
        embeddings = model.encode(batch_samples)
        embeddings_dataset[i:i+batch_size] = embeddings

        print(f"Processed batch {i // batch_size + 1}/{num_samples // batch_size + 1}")


