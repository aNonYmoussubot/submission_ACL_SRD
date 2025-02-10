import json
import h5py
import re
from nltk.stem import PorterStemmer


stemmer = PorterStemmer()
import os
import random
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
embed_model_path = 'gte-large'

model = SentenceTransformer(embed_model_path)
workspace='xxx'
src_path = f'{workspace}/analyse_distribution_shift/embeds/kp20k_train_kp_emb.h5'
import torch
import torch.nn.functional as F
from tqdm import tqdm
with h5py.File(src_path, 'r') as f1:
    embeddings_1 = f1['embeddings'][:]
from tqdm import tqdm

def load_data(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

def jaccard_similarity(kx, ks):
    intersection = len(set(kx).intersection(set(ks)))
    union = len(set(kx).union(set(ks)))
    return intersection / union if union != 0 else 0.0

def embedding_similarity(query_id, candidate_id):
    embedding_kx = embeddings_1[query_id]
    embedding_ks = embeddings_1[candidate_id]   

    return cosine_similarity([embedding_kx], [embedding_ks])[0][0]

def remove_hyphens_and_stem(keywords):
    cleaned_keywords = [re.sub(r'\-', '', word) for word in keywords] 
    return [stemmer.stem(word) for word in cleaned_keywords]


def calculate_rel(x_keywords, s_keywords, sem_value, alpha=0.7):
    kx = " ".join(x_keywords).lower().split() 
    ks = " ".join(s_keywords).lower().split() 
    kx_stemmed = remove_hyphens_and_stem(kx)
    ks_stemmed = remove_hyphens_and_stem(ks)

    jaccard_sim = jaccard_similarity(kx_stemmed, ks_stemmed)
    embed_sim = sem_value

    rel_score = alpha * embed_sim + (1 - alpha) * jaccard_sim
    return rel_score

def split_data(data, validation_size=3000, query_fraction=0.2):
    random.shuffle(data)
    validation_set = data[:validation_size]
    remaining_data = data[validation_size:]
    split_index = int(len(remaining_data) * query_fraction)
    query_set = remaining_data[:split_index]
    candidate_set = remaining_data[split_index:]
    
    return validation_set, query_set, candidate_set


def compute_top_scores(data_name, query_set, candidate_set, embeddings_1, batch_size_x=128, batch_size_y=16384):

    top_max_values_path = f'{workspace}/semantic_scores/{data_name}_top_value.pt'
    top_min_values_path = f'{workspace}/semantic_scores/{data_name}_min_value.pt'
    top_max_indices_path = f'{workspace}/semantic_scores/{data_name}_top_index.pt'
    top_min_indices_path = f'{workspace}/semantic_scores/{data_name}_min_index.pt'
    if os.path.exists(top_max_values_path):
        top_max_values = torch.load(top_max_values_path)
        top_max_indices = torch.load(top_max_indices_path)
        top_min_values = torch.load(top_min_values_path)
        top_min_indices = torch.load(top_min_indices_path)
    else:
        query_vectors = []
        candidate_vectors = []
        for item in query_set:
            idx = item['idx']
            query_vector = embeddings_1[idx]
            query_vectors.append(query_vector)
            
        for item in candidate_set:
            idx = item['idx']
            candidate_vector = embeddings_1[idx]
            candidate_vectors.append(candidate_vector)

        top_max_values = []
        top_max_indices = []
        top_min_values = []
        top_min_indices = []
        for i in tqdm(range(0, len(query_vectors), batch_size_x)):
            query_batch = query_vectors[i:i+batch_size_x]
            batch_scores = []

            for j in tqdm(range(0, len(candidate_vectors), batch_size_y)):
                can_batch = candidate_vectors[j:j+batch_size_y]
                X = torch.tensor(np.array(query_batch)).cuda()
                Y = torch.tensor(np.array(can_batch)).cuda()
                batch_score = F.cosine_similarity(X.unsqueeze(1), Y.unsqueeze(0), dim=2).cpu()
                batch_scores.append(batch_score)
            full_batch_scores = torch.cat(batch_scores, dim=1)

            max_values, max_indices = torch.topk(full_batch_scores, 1000, dim=1, largest=True, sorted=True)
            top_max_values.append(max_values)
            top_max_indices.append(max_indices)

            min_values, min_indices = torch.topk(full_batch_scores, 1000, dim=1, largest=False, sorted=True)
            top_min_values.append(min_values)
            top_min_indices.append(min_indices)

        top_max_values = torch.cat(top_max_values, dim=0)
        top_max_indices = torch.cat(top_max_indices, dim=0)
        top_min_values = torch.cat(top_min_values, dim=0)
        top_min_indices = torch.cat(top_min_indices, dim=0)
        torch.save(top_max_values, top_max_values_path)
        torch.save(top_max_indices, top_max_indices_path) 
        torch.save(top_min_values, top_min_values_path)
        torch.save(top_min_indices, top_min_indices_path)           
    scores = [top_max_values, top_max_indices, top_min_values, top_min_indices]
    query_set = select_samples(scores, query_set, candidate_set)

    return query_set




def select_samples(scores, query_set, candidate_set, top_positive=10, hardest_negative=50):
    # for positive samples
    for i, query in tqdm(enumerate(query_set), desc='generate positive and negtive samples...'):
        query_keywords = query['present_kp'] + query['absent_kp']
        query_id = query['idx']
        pos_scores = []
        pos_value = scores[0][i]
        pos_index = scores[1][i]
        for j, (p_value, p_index) in enumerate(zip(pos_value, pos_index)):
            candidate_keywords = candidate_set[p_index]['present_kp'] + candidate_set[p_index]['absent_kp']
            rel_score = calculate_rel(query_keywords, candidate_keywords,p_value)
            pos_scores.append((rel_score.item(), p_index.item())) 

        high_score_indices = [p_index for rel_score, p_index in pos_scores if rel_score > 0.7]
        
        pos_candidates = [candidate_set[idx] for idx in high_score_indices]
        postive_index = high_score_indices[:top_positive]
        
        # for negative 
        neg_value = scores[2][i]
        neg_index = scores[3][i]
        neg_scores = []
        for j, (n_value, n_index) in enumerate(zip(neg_value, neg_index)):
            candidate_keywords = candidate_set[n_index]['present_kp'] + candidate_set[n_index]['absent_kp']
            rel_score = calculate_rel(query_keywords, candidate_keywords,n_value)
            neg_scores.append((rel_score.item(), n_index.item()))  
        neg_scores.sort(key=lambda x: x[0])  
        hard_neg_index = [neg_scores[idx][1] for idx in range(min(hardest_negative, len(neg_scores)))]  # 取前50个最低分数的索引
        pass   
        query_set[i]['pos_index'] = postive_index
        query_set[i]['neg_index'] = hard_neg_index

    
    return query_set

def process_data(file_path, domain_path):
    data = load_data(file_path)
    domain_data = load_data(domain_path)
    ood_list = list(range(len(domain_data)))
    validation_set, query_set, candidate_set = split_data(data)
    validation_set_path = f'{workspace}/semantic_scores/validation.json'
    query_set_path = f'{workspace}/semantic_scores/query.json'
    can_set_path = f'{workspace}/semantic_scores/candidate.json'
    if not os.path.exists(query_set_path):
        with open(query_set_path, 'w') as f:
            json.dump(query_set, f, indent=4)
    else:
        with open(query_set_path, "r", encoding="utf-8") as file:
            query_set = json.load(file)
    if not os.path.exists(can_set_path):
        with open(can_set_path, 'w') as f:
            json.dump(candidate_set, f, indent=4)
    else:
        with open(can_set_path, "r", encoding="utf-8") as file:
            candidate_set = json.load(file)
    if not os.path.exists(validation_set_path):
        with open(validation_set_path, 'w') as f:
            json.dump(validation_set, f, indent=4)
    else:
        with open(validation_set_path, "r", encoding="utf-8") as file:
            validation_set = json.load(file)            
    if not os.path.exists(output_file1):
        query_set = compute_top_scores('query', query_set[:4000], candidate_set, embeddings_1)
        pointer = 0
        for q in query_set:
            ood1 = ood_list[pointer]
            pointer = (pointer + 1) % len(ood_list)
            ood2 = ood_list[pointer] 
            pointer = (pointer + 1) % len(ood_list)
            q['ood_index'] = [ood1, ood2]
        save_results(query_set, output_file1)
    if not os.path.exists(output_file2):
        validation_set = compute_top_scores('valiation', validation_set[:600], candidate_set, embeddings_1)
        pointer = 0
        for v in validation_set:
            ood1 = ood_list[pointer]
            pointer = (pointer + 1) % len(ood_list)
            ood2 = ood_list[pointer] 
            pointer = (pointer + 1) % len(ood_list)
            v['ood_index'] = [ood1, ood2]
        save_results(validation_set, output_file2)
       

def save_results(results, output_file):
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=4)        

file_path = f'{workspace}/kp20k_clean.json'  
domain_path = f'{workspace}/valid_corpus.json'  
output_file1 = f'{workspace}/semantic_scores/query_candidate.json'  
output_file2 = f'{workspace}/semantic_scores/validation_candidate.json' 
process_data(file_path, domain_path)



