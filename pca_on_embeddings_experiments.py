# https://scikit-learn.org/stable/auto_examples/decomposition/plot_incremental_pca.html

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import json
import glob
from sklearn.decomposition import PCA, IncrementalPCA
from json_embedding_parser import JSONEmbeddingParser
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import euclidean_distances, cosine_similarity


def timestamp():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


# print("=== ignore this ===")
# scaler = StandardScaler()
# parser = JSONEmbeddingParser()
# print(f"PCA calculation on embeddings started at {timestamp()}")
# # data = parser.parse_and_embed("data_with_embeddings/scsb_nypl_embeddings_1.json")
# embeddings_matrix_path = "embeddings_matrix/embeddings_batch_1_marcxml_matrix.csv"
# # Read embedding matrix using pandas
# embedding_df = pd.read_csv(embeddings_matrix_path)
# embedding_values = embedding_df.values
# data_rescaled = scaler.fit_transform(embedding_values)  # original data matrix

# pca = PCA().fit(data_rescaled)
# cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
# n_components_95 = np.argmax(cumulative_variance >= 0.95) + 1
# print(
#     f"Number of components for 95% variance: {n_components_95}"
# )  # number of components for 95% variance: 221
# print(f"PCA calculation on embeddings completed at {timestamp()}")
# print("=== ignore this ===")

# print(
#     f"{timestamp()} Loading similarities matrix from 'similarities_matrix/similarities_incremental_03032026_matrix.csv'..."
# )
# similarities_df = pd.read_csv(
#     "similarities_matrix/similarities_incremental_03032026_matrix.csv", index_col=0
# )
# print(f"{timestamp()} Similarities matrix loaded.")
# print(f"Shape of similarities matrix: {similarities_df.shape}")
# X = similarities_df.values
# print(X[:5, :5])  # print first 5 rows and columns of the similarities matrix

## IncrementalPCA (IPCA) on embeddings ##
print(f"{timestamp()} Starting IncrementalPCA on embeddings...")
batch_size = 10000
n_components = 62  # Set as needed
# batch_files = sorted(glob.glob("embeddings_matrix/embeddings_batch_*_matrix.csv"))
batch_files = sorted(glob.glob("embeddings_matrix/scsb_update_batch_*_matrix.csv"))
ipca = IncrementalPCA(n_components=n_components, batch_size=batch_size)
scaler = StandardScaler()

for batch_file in batch_files:
    print(f"Fitting IPCA on {batch_file}")
    batch_data = pd.read_csv(batch_file).values
    batch_data = scaler.fit_transform(batch_data)
    ipca.partial_fit(batch_data)
    # scaler_batch_data = scaler.fit_transform(batch_data)
    # ipca.partial_fit(scaler_batch_data)

# After fitting, transform all embeddings and combine
transformed_batches = []
for batch_file in batch_files:
    print(f"Transforming {batch_file}")
    batch_data = pd.read_csv(batch_file).values
    batch_data = scaler.fit_transform(batch_data)
    x_ipca_batch = ipca.transform(batch_data)
    transformed_batches.append(x_ipca_batch)
    # scaler_batch_data = scaler.transform(batch_data)
    # X_ipca_batch = ipca.transform(scaler_batch_data)
    # transformed_batches.append(X_ipca_batch)

X_ipca = np.vstack(transformed_batches)
print("Final IPCA shape:", X_ipca.shape)
print("IPCA transformed data (first 5 rows):")
print(X_ipca[:5])
print("Explained variance ratio (IPCA):")
print(ipca.explained_variance_ratio_)

print("Cumulative explained variance (IPCA):")
print(np.cumsum(ipca.explained_variance_ratio_))

distances_ipca = euclidean_distances(X_ipca)
print("Sample pairwise distances in IPCA space (first 5):", distances_ipca[0, 1:6])
print("Max distance in IPCA space:", np.max(distances_ipca))
print(
    "Min distance in IPCA space (excluding zero):",
    np.min(distances_ipca[distances_ipca > 0]),
)

# print("============= Cosine similarity for PCA components =====================")
# # --- Cosine similarity for PCA components ---
# print("Calculating cosine similarity matrix for PCA components...")
# cos_sim_pca = cosine_similarity(X_ipca)
# print("Cosine similarity matrix shape:", cos_sim_pca.shape)
# print("Sample cosine similarities (first row, first 5):", cos_sim_pca[0, 1:6])
# print(
#     "Max cosine similarity (excluding self):",
#     np.max(cos_sim_pca[np.eye(cos_sim_pca.shape[0]) == 0]),
# )
# print("Min cosine similarity:", np.min(cos_sim_pca[np.eye(cos_sim_pca.shape[0]) == 0]))
# # --- Find duplicates using cosine similarity for PCA components ---
# cosine_threshold = 0.975  # Adjust as needed
# duplicate_cosine_pairs = []
# for i in range(cos_sim_pca.shape[0]):
#     for j in range(i + 1, cos_sim_pca.shape[1]):
#         if cos_sim_pca[i, j] > cosine_threshold:
#             duplicate_cosine_pairs.append((i, j))
# print(
#     f"Found {len(duplicate_cosine_pairs)} potential duplicate pairs using cosine similarity (>{cosine_threshold}) in PCA space."
# )
# if duplicate_cosine_pairs:
#     print("Duplicate pair indices and record IDs (cosine):")
#     combined_ids = []
#     batch_json_files = sorted(
#         glob.glob("data_with_embeddings/scsb_update_*_batch_1.json")
#     )
#     for batch_json_file in batch_json_files:
#         with open(batch_json_file, "r") as f:
#             batch_data = json.load(f)
#             combined_ids.extend(
#                 [item.get("id", f"index_{idx}") for idx, item in enumerate(batch_data)]
#             )
#     for i, j in duplicate_cosine_pairs:
#         id_i = combined_ids[i] if i < len(combined_ids) else f"index_{i}"
#         id_j = combined_ids[j] if j < len(combined_ids) else f"index_{j}"
#         print(f"Pair: ({i}, {j}) -> IDs: {id_i}, {id_j}")
# print("============ end of Cosine similarity for PCA components ======================")


# Find duplicates in IPCA space
# Max distance in IPCA space: 1.3711704915933614
# Min distance in IPCA space (excluding zero): 0.013151091754532652

threshold_ipca = 2.39  # Adjust this value

print("threshold_ipca:", threshold_ipca)
duplicate_ipca_pairs = []
for i in range(distances_ipca.shape[0]):
    for j in range(i + 1, distances_ipca.shape[1]):
        if distances_ipca[i, j] < threshold_ipca:
            duplicate_ipca_pairs.append((i, j))
print(f"Found {len(duplicate_ipca_pairs)} potential duplicate pairs in IPCA space.")
# Build combined ID list from all batch JSON files
combined_ids = []
batch_json_files = sorted(glob.glob("data_with_embeddings/scsb_update_*_batch_1.json"))
for batch_json_file in batch_json_files:
    with open(batch_json_file, "r") as f:
        batch_data = json.load(f)
        combined_ids.extend(
            [item.get("id", f"index_{idx}") for idx, item in enumerate(batch_data)]
        )
if duplicate_ipca_pairs:
    print("Duplicate pair indices and record IDs (IPCA):")
    for i, j in duplicate_ipca_pairs:
        id_i = combined_ids[i] if i < len(combined_ids) else f"index_{i}"
        id_j = combined_ids[j] if j < len(combined_ids) else f"index_{j}"
        print(f"Pair: ({i}, {j}) -> IDs: {id_i}, {id_j}")

### start PCA ###
# print(f"{timestamp()} Starting standard PCA fit...")
# print(f"Shape of X before standard PCA: {X.shape}")
# print(f"{timestamp()} Running standard PCA with n_components={n_components}...")
# pca = PCA(n_components=n_components)
# X_pca = pca.fit_transform(X)
# print(f"{timestamp()} Standard PCA fit complete.")

# # Plotting (no target labels, so just scatter all points)
# for X_transformed, title in [(X_pca, "PCA")]:
#     print(f"Plotting results for {title}...")
#     plt.figure(figsize=(8, 8))
#     plt.scatter(X_transformed[:, 0], X_transformed[:, 1], color="navy", lw=2)
#     plt.title(title + " on similarities_incremental_03032026_matrix.csv")
#     plt.axis("equal")

# # X_pca is the transformed data
# print("PCA transformed data (first 5 rows):")
# print(X_pca[:5])

# # Explained variance ratio for each component
# print("Explained variance ratio:")
# print(pca.explained_variance_ratio_)

# # Cumulative explained variance
# print("Cumulative explained variance:")
# print(np.cumsum(pca.explained_variance_ratio_))

# # Plot cumulative explained variance for all components
# # The elbow point of the plot indicates the the number of components to retain for a good balance between dimensionality reduction and information retention
# plt.figure(figsize=(10, 6))
# pca_full = PCA().fit(X)
# cumulative_variance = np.cumsum(pca_full.explained_variance_ratio_)
# plt.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, marker="o")
# plt.xlabel("Number of Components")
# plt.ylabel("Cumulative Explained Variance")
# plt.title("Cumulative Explained Variance by Number of PCA Components")
# plt.grid(True)
# plt.tight_layout()

# # Calculate distances of PCA components
# distances = euclidean_distances(
#     X_pca
# )  # what else can I use to calculate the distance? cosine similarity?
# print("Sample pairwise distances (first 5):", distances[0, 1:6])
# print("Max distance:", np.max(distances))
# print("Min distance (excluding zero):", np.min(distances[distances > 0]))

# ## Find duplicates. Change the threshold until we find a reasonable amount of duplicates ##
# # X_pca: shape (n_samples, n_components)
# distances = euclidean_distances(
#     X_pca
# )
# duplicate_pairs = []
# threshold = 0.74  # Adjust this value
# for i in range(distances.shape[0]):
#     for j in range(i + 1, distances.shape[1]):
#         if distances[i, j] < threshold:
#             duplicate_pairs.append((i, j))
# print(f"Found {len(duplicate_pairs)} potential duplicate pairs in PCA space.")
# if duplicate_pairs:
#     print("Duplicate pair indices and record IDs:")
#     # Load original data to get record IDs


#     with open("fixed_json/incremental_fixed_03032026.json", "r") as f:
#         original_data = json.load(f)
#     for i, j in duplicate_pairs:
#         id_i = original_data[i].get("id", f"index_{i}")
#         id_j = original_data[j].get("id", f"index_{j}")
#         print(f"Pair: ({i}, {j}) -> IDs: {id_i}, {id_j}")

# Calculate and print the number of components needed for 95% explained variance
# threshold = 0.95
# n_components_95 = np.argmax(cumulative_variance >= threshold) + 1
# print(
#     f"Number of components needed for {int(threshold * 100)}% explained variance: {n_components_95}"
# )  # number of components: 44
########## end PCA ###########

# # start Cosine similarity for PCA-transformed data
# print("Calculating cosine similarity matrix for PCA-transformed data...")
# cos_sim = cosine_similarity(X_pca)
# print("Cosine similarity matrix shape:", cos_sim.shape)
# print("Sample cosine similarities (first row, first 5):", cos_sim[0, 1:6])
# print("Max cosine similarity (excluding self):", np.max(cos_sim[np.eye(cos_sim.shape[0]) == 0]))
# print("Min cosine similarity:", np.min(cos_sim[np.eye(cos_sim.shape[0]) == 0]))

# # Find duplicate pairs using cosine similarity
# cosine_threshold = 0.99  # Adjust as needed
# duplicate_cosine_pairs = []
# for i in range(cos_sim.shape[0]):
#     for j in range(i + 1, cos_sim.shape[1]):
#         if cos_sim[i, j] > cosine_threshold:
#             duplicate_cosine_pairs.append((i, j))
# print(f"Found {len(duplicate_cosine_pairs)} potential duplicate pairs using cosine similarity (>{cosine_threshold}) in PCA space.")
# if duplicate_cosine_pairs:
#     print("Duplicate pair indices and record IDs (cosine):")
#     import json
#     with open("fixed_json/incremental_fixed_03032026.json", "r") as f:
#         original_data = json.load(f)
#     for i, j in duplicate_cosine_pairs:
#         id_i = original_data[i].get("id", f"index_{i}")
#         id_j = original_data[j].get("id", f"index_{j}")
#         print(f"Pair: ({i}, {j}) -> IDs: {id_i}, {id_j}")
# print("Min cosine similarity:", np.min(cos_sim[np.eye(cos_sim.shape[0]) == 0]))
###### end Cosine similarity ###########

print("Sample pairwise distances in IPCA space (first 5):", distances_ipca[0, 1:6])
print("Max distance in IPCA space:", np.max(distances_ipca))
print(
    "Min distance in IPCA space (excluding zero):",
    np.min(distances_ipca[distances_ipca > 0]),
)

print("Showing plots...")

plt.show()
