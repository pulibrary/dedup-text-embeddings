from json_embedding_parser import JSONEmbeddingParser
import json
import glob
import pandas as pd
import numpy as np
import os


def main():

    parser = JSONEmbeddingParser()
    # Process embeddings in batches and save results
    all_embedded = parser.parse_and_embed(
        "fixed_json/large_data.json", batch_size=10000
    )

    # For each batch matrix CSV, calculate similarities and find duplicates
    batch_matrix_files = sorted(
        glob.glob("embeddings_matrix/embeddings_batch_*_matrix.csv")
    )
    for batch_idx, batch_matrix_file in enumerate(batch_matrix_files):
        print(f"Processing batch matrix: {batch_matrix_file}")
        embeddings_matrix = pd.read_csv(batch_matrix_file).values
        print("Embeddings matrix shape:", embeddings_matrix.shape)

        # # Calculate similarities
        # model = parser.model
        # similarities = model.similarity(embeddings_matrix, embeddings_matrix)
        # n = similarities.shape[0]
        # # col_names = [f"embd_{i + 1}" for i in range(n)]
        # # row_names = [f"doc_{i + 1}" for i in range(n)]
        # sim_csv_path = (
        #     f"similarities_matrix/similarities_batch_{batch_idx + 1}_matrix.csv"
        # )
        # os.makedirs("similarities_matrix", exist_ok=True)
        # df = pd.DataFrame(similarities)
        # df.to_csv(sim_csv_path, index=False, header=False)
        # print(f"Saved similarities matrix: {sim_csv_path}")

        # # Find duplicates
        # duplicates = parser.find_duplicates(similarities, threshold=0.95)
        # print(
        #     f"Number of duplicate pairs found in batch {batch_idx + 1}: {len(duplicates)}"
        # )
        # # Optionally print duplicate pairs
        # batch_json_path = f"data_with_embeddings/embeddings_batch_{batch_idx + 1}.json"
        # with open(batch_json_path, "r") as f:
        #     batch_data = json.load(f)
        # for i, j in duplicates:
        #     id_i = batch_data[i].get("id", f"index_{i}")
        #     id_j = batch_data[j].get("id", f"index_{j}")
        #     print(f"Duplicate pair: {id_i} and {id_j}")


if __name__ == "__main__":
    main()
