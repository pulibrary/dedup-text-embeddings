# Identify similarity and duplicates using text embeddings and PCA
-- copy of [dedup_pca](https://github.com/christinach/dedup_pca) 
## Text Embeddings
In this project we use the [SentenceTransformers (SBERT)](https://sbert.net/) to create text embeddings from the following marc fields:

- id: 001,  
- title: 245$a, 245$p, 245$f  
- title inclusive dates: 245$b
- pagination: 300$a  
- publication year: 008[7:11], 008[11:15], 264$c, 260$c
- vernacular title: linked fields of 245
- vernacular author: linked fields of 100, 110, 111
- context title: 505$t
- edition: 250$a
- publisher name: 264$b, 260$b

## Methodology
As a first step we use the trained model [all-MiniLM-L6-v2](https://huggingface.co/nreimers/MiniLM-L6-H384-uncased) which according to the [documentation it is 5 times faster and still offers good quality](https://sbert.net/docs/sentence_transformer/pretrained_models.html). 

We parse the MARCXML file and create a string from the values of the MARC fields. We use this string to create text embeddings. We're saving the text embedding in a new JSON file that includes the text_embedding field and the record id. 


## Run the program
In order to test the program:
- Clone the repo 
- Install jupyter notebooks in your local environment `pip install jupyterlab`   
- Go to the [Bibdata evens page](https://bibdata.princeton.edu/events). Find the events with the label `partner updates`. Download a few dump files. Don't use files that have delete in the dump file name. Rename the `...xml.gz` files so that they match the following naming convention `scsb_update_*.xml.gz`. Save the files in the directory `data_marcxml`.
- In your terminal run `jupyter lab`. This will load jupyter notebooks in localhost. Find and select the notebook file `dedup_embeddings_pca.ipynb`. Select `Kernel` -> `Restart Kernel and Run all cells`.

## Next steps
1. Use different sentence transformers models to produce text-embeddings. Apply PCA and compare the results.
2. Use a clustering method to cluster the resulting pairs from the PCA components that have the same id.
3. Use K-means instead of PCA to cluster the text-embeddings. See [Semantic Deduplication](https://docs.nvidia.com/nemo/curator/latest/curate-text/process-data/deduplication/semdedup.html) for different methods.
4. Write the repo in Rust. Use [sklears](https://docs.rs/sklears/latest/sklears/) and [polars](https://docs.rs/polars/latest/polars/)

## References
1. [Principal component Analysis](https://www.nature.com/articles/nmeth.4346)
1. [An intuitive introduction to text embeddings](https://stackoverflow.blog/2023/11/09/an-intuitive-introduction-to-text-embeddings/)
2. [SentenceTransformers](https://sbert.net/docs/installation.html)
3. [Why is it ok to average embeddings?](https://randorithms.com/2020/11/17/Adding-Embeddings.html)
3. [Deep Learning vs Principal Component Analysis](https://medium.com/@abatrek059/deep-learning-vs-principal-component-analysis-a-comparative-example-0a9bb375c8bb)
4. [How to train Sentence Transformers](https://github.com/huggingface/blog/blob/main/how-to-train-sentence-transformers.md)
5. [Linking Theory and Practice of Digital Libraries](https://link.springer.com/book/10.1007/978-3-031-16802-4)
6. [Mastering Text Embeddings: A Key Ingredient for RAG Success](https://sukalp.medium.com/mastering-text-embeddings-a-key-ingredient-for-rag-success-1a3ed01beb56#:~:text=Normalization%20ensures%20that%20embeddings%20are,crucial%20for%20accurate%20similarity%20comparisons.)
7. [Semantic Deduplication](https://docs.nvidia.com/nemo/curator/latest/curate-text/process-data/deduplication/semdedup.html)