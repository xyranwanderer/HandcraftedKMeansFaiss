# HandcraftedKMeansFaiss
A hybrid clustering pipeline combining handcrafted K-Means with Faiss for efficient high-dimensional vector clustering and neighbor search.

This repository contains Python scripts for PCA dimensionality reduction, clustering, and merging clusters for large-scale vector data. The scripts utilize FAISS, scikit-learn, and Dask to process and analyze vector data efficiently.

## Files in the Repository

1. **faiss-01.py**  
   - **Purpose**: Computes the required number of principal components to retain a specified variance (`target_variance`) in the data using PCA.  
   - **Key Features**:
     - Uses Dask to handle large-scale data with efficient chunking.
     - Outputs the optimal number of principal components for subsequent dimensionality reduction.  
   - **Usage**:
     ```bash
     python faiss-01.py
     ```
     Update the `DATA_DIR` and `OUTPUT_FILE` variables in the script to point to your input vector directory and output file path.

2. **faiss-02.py**  
   - **Purpose**: Performs PCA dimensionality reduction based on the number of components determined in `faiss-01.py`.  
   - **Key Features**:
     - Outputs reduced-dimension vectors and saves the PCA model for later use.  
   - **Usage**:
     ```bash
     python faiss-02.py
     ```
     Configure `data_dir`, `output_file`, and `pca_model_file` in the script with appropriate paths.

3. **faiss-03.py**  
   - **Purpose**: Performs K-Means clustering on PCA-reduced vectors.  
   - **Key Features**:
     - Clusters the data into a specified number of groups.
     - Saves cluster results and vector mappings to files.  
   - **Usage**:
     ```bash
     python faiss-03.py
     ```
     Ensure paths for `PCA_FILE`, `OUTPUT_DIR`, and `full_bid_lis` are correctly set.

4. **faiss-04.py**  
   - **Purpose**: Conducts FAISS-based similarity search on clustered data.  
   - **Key Features**:
     - Maps similar vectors within clusters using cosine similarity.
     - Saves clustering results to HDF5 and pickle files.  
   - **Usage**:
     ```bash
     python faiss-04.py
     ```
     Update `CLUSTER_DIR` and `OUTPUT_FILE` with the appropriate paths.

5. **faiss-045merge.py**  
   - **Purpose**: Merges sub-clusters generated in previous steps to create larger clusters.  
   - **Key Features**:
     - Implements a graph-based approach to find connected components.
     - Outputs merged clusters in HDF5 and pickle formats.  
   - **Usage**:
     ```bash
     python faiss-045merge.py
     ```
     Specify `INPUT_DIR` and `OUTPUT_DIR` paths.

## Setup and Installation

1. Clone the repository:
   ```bash
   git clone <repository_url>
   cd <repository_directory>

