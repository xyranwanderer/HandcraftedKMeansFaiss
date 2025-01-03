import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import numpy as np
import json
import logging
from sklearn.decomposition import PCA
import pickle

def pca_reduction(data_dir, output_file, pca_model_file, n_components=355):
    try:
        print("Start loading vectors from vector files...")
        mtx_lis = sorted([k for k in os.listdir(data_dir) if k.startswith('DFN5B') and k.endswith(".npy")])
        idx_lis = sorted([k for k in os.listdir(data_dir) if k.startswith('DFN5B') and k.endswith(".txt")])
        print("Finish loading vectors...")

        full_bid_mtx_lis = []
        full_bid_lis = []

        print("Start loading vectors...")
        for idx_f, mtx_f in zip(idx_lis, mtx_lis):
            full_bid_mtx_lis.extend(np.load(os.path.join(data_dir, mtx_f)))
            with open(os.path.join(data_dir, idx_f), 'r') as f:
                full_bid_lis.extend(json.loads(f.read()))
        print("Finish loading vectors")

        vectors = np.array(full_bid_mtx_lis, dtype="float32")
        n_vectors, dim = vectors.shape
        print(f"number of vectors: {n_vectors}, dim: {dim}")

        print("Start PCA...")
        pca = PCA(n_components=n_components)
        vectors_reduced = pca.fit_transform(vectors) 

        np.save(output_file, vectors_reduced)
        print(f"Save PCA result to {output_file}")

        # Save PCA model as .pkl file
        with open(pca_model_file, "wb") as f:
            pickle.dump(pca, f)
        print(f"Save PCA model to {pca_model_file}")
    except Exception as e:
        print(f"An error occurred during PCA dimensionality reduction: {e}")

if __name__ == "__main__":
    data_dir = "/path/to/your/vector/"
    output_file = "/path/to/save/your/pca_reduced.npy" # Save Pca model as numpy file
    pca_model_file = "/path/to/save/your/pca_reduced_095_model.pkl"  # Save Pca model as Pickle file
    pca_reduction(data_dir, output_file, pca_model_file, n_components=355) # Change the `n_components` to what you got from `faiss-01.py`