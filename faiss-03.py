import os
import numpy as np
import json
import logging
from sklearn.cluster import KMeans
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

def load_full_bid_lis(pkl_file):
    try:
        logging.info(f"Start loading {pkl_file}...")
        with open(pkl_file, "rb") as f:
            full_bid_lis = pickle.load(f)
        logging.info(f"Finish loading {pkl_file}")
        return full_bid_lis
    except Exception as e:
        logging.error(f"There is an error when loading {pkl_file} : {e}")
        return None

def kmeans_clustering(pca_file, output_dir, full_bid_lis):
    try:
        logging.info("Start loading PCA dimensionality reduction results...")
        vectors_reduced = np.load(pca_file)
        logging.info("Finish loading PCA dimensionality reduction results")

        logging.info("Start K-Means clustering...")
        num_clusters = 50
        kmeans = KMeans(n_clusters=num_clusters, random_state=42)
        labels = kmeans.fit_predict(vectors_reduced)
        logging.info("Finish K-Means clustering")

        for i in range(num_clusters):
            cluster_indices = np.where(labels == i)[0]  
            cluster_vectors = vectors_reduced[cluster_indices]
            cluster_names = [full_bid_lis[idx] for idx in cluster_indices]  
            cluster_size = np.sum(labels == i)
            logging.info(f"The size if Cluster {i} is: {cluster_size}")

            np.save(os.path.join(output_dir, f"cluster_{i}_vectors.npy"), cluster_vectors)

        logging.info(f"All clustering results have been saved to {output_dir}")
    except Exception as e:
        logging.error(f"Error occurred while K-Means clustering: {e}")

if __name__ == "__main__":
    PCA_FILE = "/path/to/your/pca_reduced_095.npy"
    OUTPUT_DIR = "/path/to/save/your/vec_id_mapping_files/"

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    full_bid_lis = load_full_bid_lis("/data1/users/xxx/full_bid_lis.pkl") # This is the original names of your vectors, you need to save it seperately

    if full_bid_lis is not None:
        kmeans_clustering(PCA_FILE, OUTPUT_DIR, full_bid_lis)
    else:
        logging.error("cannot load full_bid_lis，terminated!")