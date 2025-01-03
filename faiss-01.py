import os
import numpy as np
import json
import logging
from sklearn.decomposition import PCA
import pickle

import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)

import os
import numpy as np
import json
import logging
import dask.array as da
from dask_ml.decomposition import PCA as DaskPCA

def pca_reduction(data_dir, output_file, target_variance=0.95):
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

        # change data to Dask array
        vectors_dask = da.from_array(vectors, chunks=(10000, dim))  # Chunking

        print("Start calculating the variance contribution rates of all principal components...")
        pca_full = DaskPCA(n_components=dim)  # set n_components as original dim
        pca_full.fit(vectors_dask)

        explained_variance_ratio = pca_full.explained_variance_ratio_
        cumulative_variance = np.cumsum(explained_variance_ratio)

        n_components = np.argmax(cumulative_variance >= target_variance) + 1
        print(f"The number of principal components to retain: {n_components}")
    except Exception as e:
        print(f"An error occurred during PCA dimensionality reduction: {e}")

if __name__ == "__main__":
    DATA_DIR = "/path/to/your/vector/"
    OUTPUT_FILE = "/path/to/save/your/pca_reduced.npy"
    target_variance=0.95

    pca_reduction(DATA_DIR, OUTPUT_FILE, target_variance)