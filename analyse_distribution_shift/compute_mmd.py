import h5py
import numpy as np
from sklearn.metrics.pairwise import rbf_kernel
src_path = 'embeds/kp20k_train_emb.h5'

from tqdm import tqdm
with h5py.File(src_path, 'r') as f1:
    embeddings_1 = f1['embeddings'][:]

def compute_mmd(X, Y, sigma=1.0):

    K_xx = rbf_kernel(X, X, gamma=1.0 / (2 * sigma ** 2))
    K_yy = rbf_kernel(Y, Y, gamma=1.0 / (2 * sigma ** 2))
    K_xy = rbf_kernel(X, Y, gamma=1.0 / (2 * sigma ** 2))

    mmd_squared = np.mean(K_xx) + np.mean(K_yy) - 2 * np.mean(K_xy) 
    return mmd_squared

def batch_compute_mmd(X, Y, batch_size=1000, sigma=0.5):
    mmd_total = 0
    num_batches = len(X) // batch_size
    Y_batch = Y[:]
    for i in tqdm(range(num_batches + 1)):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, len(X))
        X_batch = X[start_idx:end_idx]
        mmd_total += compute_mmd(X_batch, Y_batch, sigma)
    
    return mmd_total / (num_batches + 1)


for data_name in ["semeval", "inspec", "nus", "krapivin", "kp20k", "duc", "openkp", "kptimes", "stack", "kpbiomed"]:
    trg_path = 'embeds/{}_emb.h5'.format(data_name)
    with h5py.File(trg_path, 'r') as f2:
        embeddings_2 = f2['embeddings'][:]
    mmd_batch_distance = batch_compute_mmd(embeddings_1, embeddings_2)
    print(f"Batch kp20k and {data_name} MMD Distance: {mmd_batch_distance}")