import json
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.decomposition import PCA, IncrementalPCA
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import statsmodels.api as sm
import pandas as pd
import os

# # 1. Load a pretrained Sentence Transformer model
# model = SentenceTransformer("all-MiniLM-L6-v2")

# # The sentences to encode
# sentences = [
#     "The weather is lovely today.",
#     "It's so sunny outside!",
#     "He drove to the stadium.",
# ]

# # 2. Calculate embeddings by calling model.encode()
# embeddings = model.encode(sentences)
# print(embeddings.shape)
# [3, 384]

# # 3. Calculate the embedding similarities
# similarities = model.similarity(embeddings, embeddings)
# print(similarities)
# tensor([[1.0000, 0.6660, 0.1046],
#          [0.6660, 1.0000, 0.1411],
#          [0.1046, 0.1411, 1.0000]])


class JSONEmbeddingParser:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def parse_and_embed(self, json_path, batch_size=10000):
        with open(json_path, "r") as f:
            data = json.load(f)

        batches = [data[i : i + batch_size] for i in range(0, len(data), batch_size)]
        all_embedded = []
        for batch_idx, batch in enumerate(batches):
            print(f"Processing batch {batch_idx + 1}/{len(batches)}...")
            for entry in batch:
                fields = [
                    entry.get("title_display", ""),
                    entry.get("uniform_title_s", ""),
                    entry.get("author_s", ""),
                    entry.get("edition_display", ""),
                    entry.get("publication_display", ""),
                    entry.get("isbn_display", ""),
                    entry.get("oclc_s", []),
                    # entry.get("call_number_display", ""),
                    # entry.get("series_statement_index", ""),
                    # entry.get("context_title_index", "")
                ]
                text = " ".join(
                    [
                        field if isinstance(field, str) else " ".join(field)
                        for field in fields
                    ]
                )
                embedding = self.model.encode(text)
                entry["text_embedding"] = embedding.tolist()

            # Save batch as JSON in data_with_embeddings
            os.makedirs("data_with_embeddings", exist_ok=True)
            batch_json_path = (
                f"data_with_embeddings/embeddings_batch_{batch_idx + 1}.json"
            )
            with open(batch_json_path, "w") as f:
                json.dump(batch, f, indent=2)
            print(f"Saved batch JSON: {batch_json_path}")

            # Save batch embeddings as CSV matrix in embeddings_matrix
            os.makedirs("embeddings_matrix", exist_ok=True)
            embeddings = [entry["text_embedding"] for entry in batch]

            df = pd.DataFrame(embeddings)
            batch_csv_path = (
                f"embeddings_matrix/embeddings_batch_{batch_idx + 1}_matrix.csv"
            )
            df.to_csv(batch_csv_path, index=False)
            print(f"Saved batch matrix CSV: {batch_csv_path}")

            all_embedded.extend(batch)
        return all_embedded

    def save_embedded_json(self, data, output_path):
        with open(output_path, "w") as f:
            json.dump(data, f, indent=2)

    def get_embeddings_matrix(self, data):
        """
        Returns a numpy array of shape (num_documents, embedding_dim)
        """
        embeddings = [
            entry["text_embedding"] for entry in data if "text_embedding" in entry
        ]
        return np.array(embeddings)

    def find_duplicates(self, similarities, threshold=0.95):
        """
        Returns a list of index pairs (i, j) where similarity > threshold and i != j
        """
        duplicates = []
        for i in range(similarities.shape[0]):
            for j in range(i + 1, similarities.shape[1]):
                if similarities[i, j] > threshold:
                    duplicates.append((i, j))
        return duplicates
