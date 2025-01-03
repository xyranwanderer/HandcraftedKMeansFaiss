import os
import numpy as np
import json
import logging
import faiss
import h5py
import pickle
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)

def faiss_search(cluster_dir, output_file):
    try:
        clusters = {}
        threshold = 0.98  # the threshold of cos similarity

        pca_model_path = "/path/to/save/your/pca_reduced_095_model.pkl"  
        logging.info("Start loading PCA model...")
        with open(pca_model_path, "rb") as f:
            pca = pickle.load(f)
        logging.info("PCA model loading completed")

        # 遍历所有聚类文件
        cluster_files = sorted([f for f in os.listdir(cluster_dir) if f.endswith("_vectors.npy")]) #  the vectors in the _vectors.npy file have been dimensionally reduced, and then need to be raised back to their original dims.
        logging.info(f"Start loading clustering vector file")
        for cluster_file in tqdm(cluster_files, desc="loading clustering vector file"):
            cluster_id = int(cluster_file.split("_")[1])
            cluster_vectors = np.load(os.path.join(cluster_dir, cluster_file)) 
            cluster_ids = json.load(open(os.path.join(cluster_dir, f"cluster_{cluster_id}_names.txt"), 'r')) 

            # Check the dimensions of `cluster_vectors``
            if len(cluster_vectors.shape) != 2:
                logging.error(f"Cluster file {cluster_file} has incorrect dimensions. It should be 2 dimensions but is actually {len(cluster_vectors.shape)}.")
                continue
            logging.info(f"Dimensions of cluster file {cluster_file}: {cluster_vectors.shape}")

            # Restore cluster_vectors to original dimensions (1024 dimensions)
            logging.info(f"Start restoring the vector of {cluster_id}th cluster to its original dimensions...")
            cluster_vectors_restored = pca.inverse_transform(cluster_vectors)
            logging.info(f"The vector dimension of the restored {cluster_id}th cluster: {cluster_vectors_restored.shape}")

            vector_id_map = {tuple(cluster_vectors_restored[i].tolist()): cluster_ids[i] for i in range(len(cluster_vectors_restored))}
            

            dim = cluster_vectors_restored.shape[1]
            index = faiss.IndexFlatIP(dim)
            index.add(cluster_vectors_restored)
            k = min(5, len(cluster_vectors_restored))
            D, I = index.search(cluster_vectors_restored, k)  
            
            for j in tqdm(range(len(cluster_vectors_restored)), desc=f"searching {cluster_id}"):
                similar_indices = I[j][D[j] > threshold]  

                if len(similar_indices) > 1:  
                    vector_tuple = tuple(cluster_vectors_restored[j].tolist())
                    if vector_tuple in vector_id_map:
                        vector_id = vector_id_map[vector_tuple]
                        
                        similar_names = set()
                        for idx in similar_indices:
                            similar_vector_tuple = tuple(cluster_vectors_restored[idx].tolist())
                            if similar_vector_tuple in vector_id_map:
                                similar_names.add(vector_id_map[similar_vector_tuple])
                       
                        clusters[vector_id] = similar_names
                        
                    else:
                        logging.warning(f"Vector {vector_tuple} no corresponding name found")

            current_output_file = output_file.replace(".h5", f"_cluster_{cluster_id}.h5")
            save_clusters_to_hdf5(current_output_file, clusters, big_cluster_id=cluster_id)
            logging.info(f"Saved results of cluster file {cluster_file} to {current_output_file}")

            clusters.clear()

            del vector_id_map, index, D, I
            logging.info(f"Variable memory for cluster file {cluster_file} has been released")

        logging.info("FAISS search completed")
    except Exception as e:
        logging.error(f"An error occurred while searching for FAISS: {e}")

def save_clusters_to_hdf5(output_file, clusters, big_cluster_id=None ):
    try:
        # filter the empty clusters
        filtered_clusters = {key: cluster for key, cluster in clusters.items() if isinstance(cluster, (list, set)) and len(cluster) > 0}
        # filtered_clusters = {
        #     key: cluster for key, cluster in clusters.items() 
        #     if isinstance(cluster, (list, set)) and len(cluster) > 1
        # }

        if not filtered_clusters:
            logging.warning("There are no valid clusters and the save operation is aborted.")
            return
        else:
            logging.info(f"There are {len(filtered_clusters)} clusters in filtered_clusters")

        total_clusters = len(filtered_clusters)  
        clusters_with_multiple_vectors = 0  

        id_to_cluster_map = {}

        sub_cluster_counter = {}

        with h5py.File(output_file, "w") as f:
            for cluster_id, cluster in filtered_clusters.items():
                if len(cluster) == 0:  
                    continue

                if len(cluster) >= 2:
                    clusters_with_multiple_vectors += 1

                cluster = [str(idx) for idx in cluster]

                cluster_ids = cluster 

                sub_cluster_size = len(cluster_ids)

                if big_cluster_id not in sub_cluster_counter:
                    sub_cluster_counter[big_cluster_id] = 0
                sub_cluster_counter[big_cluster_id] += 1
                sub_cluster_id = sub_cluster_counter[big_cluster_id]

                cluster_name = f"cluster_{big_cluster_id}_{sub_cluster_size}_{sub_cluster_id}"

                f.create_dataset(
                    cluster_name,
                    data=np.array(cluster_ids, dtype="S"),  
                    compression="gzip"  
                )


                for i, id_ in enumerate(cluster_ids):
                    id_to_cluster_map[id_] = cluster_name

        pickle_file_path = output_file.replace(".h5", ".pkl")
        logging.info(f"Generated pickle file path: {pickle_file_path}")
        with open(pickle_file_path, "wb") as pkl_file:
            pickle.dump(id_to_cluster_map, pkl_file)
        logging.info(f"The reverse dictionary has been saved as a pickle file: {pickle_file_path}")

        logging.info(f"Total number of clusters: {total_clusters}")
        logging.info(f"Number of clusters containing two or more vectors: {clusters_with_multiple_vectors}")
        logging.info(f"All clustering results have been saved to {output_file}")
    except Exception as e:
        logging.error(f"An error occurred while saving clustering results: {e}")

if __name__ == "__main__":
    CLUSTER_DIR = "/data1/users/rxy/vec_id_files/" 
    OUTPUT_FILE = "/data1/users/rxy/test_output/clusters_optimized_final.h5"

    faiss_search(CLUSTER_DIR, OUTPUT_FILE)