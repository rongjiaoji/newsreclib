import torch
import pandas as pd
import numpy as np

news_tsv_path = '../data/news.tsv'  # Adjust to your data path
embeddings_path = '../data/custom_embeddings.pt'

news_df = pd.read_csv(news_tsv_path, sep='\t', header=None, 
                      names=['news_id', 'category', 'subcategory', 'title', 'abstract', 
                             'url', 'entity_title', 'entity_abstract'])

embedding_dim = 300  # set your embedding dimension
embeddings = {news_id: np.random.rand(embedding_dim) for news_id in news_df['news_id']}

# Save embeddings clearly as a dictionary mapping news_id -> embedding
torch.save(embeddings, embeddings_path)
