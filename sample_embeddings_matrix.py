import pandas as pd
import numpy as np
import glob


class SampleEmbeddingsMatrix:
    def __init__(
        self, output_path="embeddings_matrix/sample_10000_matrix.csv", sample_size=10000
    ):
        self.output_path = output_path
        self.sample_size = sample_size

    def create_sample(self):
        batch_files = sorted(
            glob.glob("embeddings_matrix/scsb_update_batch_*_matrix.csv")
        )
        print("Batch files found:", batch_files)
        sampled_rows = []
        rows_needed = self.sample_size
        for batch_file in batch_files:
            df = pd.read_csv(batch_file, skiprows=1)
            n_rows = len(df)
            if rows_needed <= 0:
                break
            n_sample = min(rows_needed, n_rows)
            sampled_rows.append(df.sample(n=n_sample, random_state=42))
            rows_needed -= n_sample
        sampled_df = pd.concat(sampled_rows, ignore_index=True)
        sampled_df = sampled_df.iloc[: self.sample_size]
        sampled_df.to_csv(self.output_path, index=False)
        print(f"Saved {len(sampled_df)} sampled rows to {self.output_path}")


if __name__ == "__main__":
    sampler = SampleEmbeddingsMatrix()
    sampler.create_sample()
