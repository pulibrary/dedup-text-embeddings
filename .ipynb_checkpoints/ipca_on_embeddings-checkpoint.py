import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import json
import glob
from sklearn.decomposition import PCA, IncrementalPCA
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import euclidean_distances
from sample_embeddings_matrix import SampleEmbeddingsMatrix


class IPCAOnEmbeddings:
    def __init__(self):
        self._embedding_values_cache = None

    @staticmethod
    def timestamp():
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def calculate_number_of_components_with_pca(self, variance_threshold=0.95):
        embedding_values = self._embedding_values()
        data_rescaled = self._data_rescaling_with_pca(embedding_values)
        pca = PCA().fit(data_rescaled)
        cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
        n_components = np.argmax(cumulative_variance >= variance_threshold) + 1
        n_features = data_rescaled.shape[1]
        n_components = min(n_components, n_features)
        print(
            f"Number of components for {variance_threshold * 100}% variance: {n_components}"
        )
        return n_components

    def _data_rescaling_with_pca(self, embedding_values):
        scaler = StandardScaler()
        data_rescaled = scaler.fit_transform(embedding_values)
        return data_rescaled

    def _embedding_values(self):
        from sklearn.impute import SimpleImputer

        if self._embedding_values_cache is None:
            create_sample_embeddings_matrix = SampleEmbeddingsMatrix()
            create_sample_embeddings_matrix.create_sample()
            embedding_df = pd.read_csv("embeddings_matrix/sample_10000_matrix.csv")
            imputer = SimpleImputer(strategy="mean")
            embedding_values = imputer.fit_transform(embedding_df.values)
            self._embedding_values_cache = embedding_values
        return self._embedding_values_cache

    ## IncrementalPCA (IPCA) on embeddings ##
    def calculate_ipca(self, n_components, batch_size):
        print(f"{self.timestamp()} Starting IncrementalPCA on embeddings...")
        # batch_files = sorted(glob.glob("embeddings_matrix/scsb_update_batch_*_matrix.csv"))
        return IncrementalPCA(n_components=n_components, batch_size=batch_size)

    def ipca_fit(self, batch_files, n_components, batch_size):
        scaler = StandardScaler()
        first_batch_data = pd.read_csv(batch_files[0]).values
        n_features = first_batch_data.shape[1]
        n_samples_first_batch = first_batch_data.shape[0]
        n_components = min(n_components, n_features, n_samples_first_batch)
        print(
            f"n_components: {n_components}, n_features: {n_features}, n_samples_first_batch: {n_samples_first_batch}"
        )
        ipca = IncrementalPCA(n_components=n_components, batch_size=batch_size)
        for batch_file in batch_files:
            print(f"Fitting IPCA on {batch_file}")
            batch_data = pd.read_csv(batch_file).values
            batch_data = scaler.fit_transform(batch_data)
            ipca.partial_fit(batch_data)
        return ipca

    def ipca_transform(self, batch_files, ipca):
        scaler = StandardScaler()
        transformed_batches = []
        for batch_file in batch_files:
            print(f"Transforming {batch_file}")
            batch_data = pd.read_csv(batch_file).values
            batch_data = scaler.fit_transform(batch_data)
            x_ipca_batch = ipca.transform(batch_data)
            transformed_batches.append(x_ipca_batch)
        return transformed_batches

    def ipca_combine_transformed_batches(self, transformed_batches, ipca):
        X_ipca = np.vstack(transformed_batches)
        print("Final IPCA shape:", X_ipca.shape)
        print("IPCA transformed data (first 5 rows):")
        print(X_ipca[:5])
        print("Explained variance ratio (IPCA):")
        print(ipca.explained_variance_ratio_)

        print("Cumulative explained variance (IPCA):")
        print(np.cumsum(ipca.explained_variance_ratio_))
        return X_ipca

    def euclidean_distances_in_ipca_space(self, X_ipca):
        distances_ipca = euclidean_distances(X_ipca)
        print(
            "Sample pairwise distances in IPCA space (first 5):", distances_ipca[0, 1:6]
        )
        print("Max distance in IPCA space:", np.max(distances_ipca))
        print(
            "Min distance in IPCA space (excluding zero):",
            np.min(distances_ipca[distances_ipca > 0]),
        )
        return distances_ipca

    def threshold_ipca(self, distances_ipca):
        # return the indices for the upper triangle of the matrix;
        # get all unique pairwise distances, excluding self-distances
        flat_distances = distances_ipca[np.triu_indices_from(distances_ipca, k=1)]
        # set the threshold to get the closest 1% of pairs.
        min_distance = np.min(flat_distances)
        threshold_ipca = min_distance + 1e-4
        print("Automated threshold_ipca min_distance:", threshold_ipca)
        return threshold_ipca

    def identify_duplicates(self, euclidean_distances_in_ipca_space, threshold_ipca):
        print("threshold_ipca:", threshold_ipca)
        duplicate_ipca_pairs = []
        for i in range(euclidean_distances_in_ipca_space.shape[0]):
            for j in range(i + 1, euclidean_distances_in_ipca_space.shape[1]):
                if euclidean_distances_in_ipca_space[i, j] < threshold_ipca:
                    duplicate_ipca_pairs.append((i, j))
        print(
            f"Found {len(duplicate_ipca_pairs)} potential duplicate pairs in IPCA space."
        )
        # Build combined ID list from all batch JSON files
        combined_ids = []
        batch_json_files = sorted(
            glob.glob("data_with_embeddings/scsb_update_*_batch_1.json")
        )
        for batch_json_file in batch_json_files:
            with open(batch_json_file, "r") as f:
                batch_data = json.load(f)
                combined_ids.extend(
                    [
                        item.get("id", f"index_{idx}")
                        for idx, item in enumerate(batch_data)
                    ]
                )
        if duplicate_ipca_pairs:
            print("Duplicate pair indices and record IDs (IPCA):")
            for i, j in duplicate_ipca_pairs:
                if i < len(combined_ids) and j < len(combined_ids):
                    id_i = combined_ids[i]
                    id_j = combined_ids[j]
                    print(f"Pair: ({i}, {j}) -> IDs: {id_i}, {id_j}")

        print(
            "Sample pairwise distances in IPCA space (first 5):",
            euclidean_distances_in_ipca_space[0, 1:6],
        )
        print("Max distance in IPCA space:", np.max(euclidean_distances_in_ipca_space))
        print(
            "Min distance in IPCA space (excluding zero):",
            np.min(
                euclidean_distances_in_ipca_space[euclidean_distances_in_ipca_space > 0]
            ),
        )
