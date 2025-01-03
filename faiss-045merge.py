import os
import h5py
import logging
from collections import defaultdict
import re
import pickle
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)

def load_clusters_from_hdf5(input_dir):
    """Load all subclusters of a large cluster from an HDF5 file"""
    clusters = defaultdict(list)
    cluster_files = [f for f in os.listdir(input_dir) if f.endswith(".h5")]
    for cluster_file in cluster_files:
        # Use regular expressions to extract group numbers
        match = re.match(r".*cluster_(\d+)\.h5", cluster_file)
        if not match:
            logging.warning(f"File name format is incorrect: {cluster_file}")
            continue
        cluster_id = int(match.group(1))
        with h5py.File(os.path.join(input_dir, cluster_file), "r") as f:
            for subcluster_name in f.keys():
                subcluster = set(x.decode() for x in f[subcluster_name][:])  
                clusters[cluster_id].append(subcluster)
        logging.info(f"Loaded sub-cluster of cluster {cluster_id}")
    return clusters

def merge_clusters_for_single_cluster(cluster_list):
    """Merge subgroups of a single large group"""
    # build graph
    graph = defaultdict(set)
    for cluster in cluster_list: 
        cluster_list_sorted = sorted(cluster) 
        for i in range(len(cluster_list_sorted)): 
            for j in range(i + 1, len(cluster_list_sorted)): 
                graph[cluster_list_sorted[i]].add(cluster_list_sorted[j])
                graph[cluster_list_sorted[j]].add(cluster_list_sorted[i])

    # connected component analysis
    visited = set()
    merged_clusters = []
    for node in graph:
        if node not in visited:
            component = [] # 用To store all nodes in the current connected component
            dfs(node, graph, visited, component)
            merged_clusters.append(set(component))
    return merged_clusters # Return the merged cluster list

def dfs(node, graph, visited, component):
    """dfs"""
    if node not in visited:
        visited.add(node)
        component.append(node)
        for neighbor in graph[node]: # Then traverse the neighbor nodes currently connected to the previous node.
            dfs(neighbor, graph, visited, component)

def save_merged_clusters_to_hdf5(merged_clusters, output_dir):
    """Save merged father cluster to HDF5 files and PKL files"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for cluster_id, cluster_list in merged_clusters.items():
        # Each father cluster is saved as a separate HDF5 file
        output_file = os.path.join(output_dir, f"merged-cluster_{cluster_id}.h5")
        id_to_cluster_map = {}  # 用于保存反转字典
        with h5py.File(output_file, "w") as f:
            # Count the size of the subgroup, used for numbering
            subcluster_count = {}
            for cluster in cluster_list:
                cluster_size = len(cluster)
                if cluster_size not in subcluster_count:
                    subcluster_count[cluster_size] = 0
                subcluster_count[cluster_size] += 1

                # Generate subgroup name
                subcluster_name = f"merged-cluster_{cluster_id}_{cluster_size}_{subcluster_count[cluster_size]}"
                f.create_dataset(subcluster_name, data=list(cluster), compression="gzip")

                # Add each name in cluster to id_to_cluster_map
                for id_ in cluster:
                    id_to_cluster_map[id_] = subcluster_name
            logging.info(f"Saved {len(cluster_list)} sub-clusters of large cluster {cluster_id} to {output_file}")

        pkl_file = output_file.replace(".h5", ".pkl")
        with open(pkl_file, "wb") as f:
            pickle.dump(id_to_cluster_map, f)
        logging.info(f"Saved reverse dictionary to {pkl_file}")

def merge_and_save_clusters(input_dir, output_dir):
    """Merge clusters and save results"""
    logging.info("Start loading subgroups...")
    clusters = load_clusters_from_hdf5(input_dir)
    logging.info("Subgroup loading completed")

    logging.info("Start merging subgroups...")
    merged_clusters = {}
    for cluster_id, cluster_list in tqdm(clusters.items(), desc="merging father cluster"):
        merged_clusters[cluster_id] = merge_clusters_for_single_cluster(cluster_list)
    logging.info("Sub-group merger completed")

    logging.info("Start saving the merged subgroup...")
    save_merged_clusters_to_hdf5(merged_clusters, output_dir)
    logging.info("The merged sub-group is saved.")

if __name__ == "__main__":
    INPUT_DIR = "/data1/users/xxx/folder"  # Input directory (to save the HDF5 file of the original subcluster)
    OUTPUT_DIR = "/data1/users/xxx/output/merged_clusters"  # Output directory (save the merged subgroup)

    # Merge groups and save results
    merge_and_save_clusters(INPUT_DIR, OUTPUT_DIR)