from typing import List
import torch
import logging
from torch import nn

class CustomNewsEncoder(nn.Module):
    def __init__(self, custom_embedding_path: str):
        super().__init__()
        self.custom_embedding_path = custom_embedding_path
        
        try:
            embeddings_dict = torch.load(custom_embedding_path)
            print(f"Custom embeddings loaded from {custom_embedding_path}")
            print(f"Custom embeddings dictionary contains {len(embeddings_dict)} entries")
            print(f"Sample embedding shape: {next(iter(embeddings_dict.values())).shape}")
            
            embeddings_tensor = torch.stack([torch.tensor(e) for e in embeddings_dict.values()])
            self.embedding = nn.Embedding.from_pretrained(embeddings_tensor, freeze=True)
            self.news_id_to_index = {news_id: idx for idx, news_id in enumerate(embeddings_dict.keys())}
            
        except Exception as e:
            raise RuntimeError(f"Failed to load embeddings from {custom_embedding_path}. Error: {str(e)}")

    def forward(self, news_ids: List[str]):
        try:
            indices = torch.tensor([self.news_id_to_index[news_id] for news_id in news_ids], 
                                 device=self.embedding.weight.device)
            embeddings = self.embedding(indices)
            
            # Validation check
            assert embeddings.shape[-1] == 300, f"Expected embedding dim 300, got {embeddings.shape[-1]}"
            return embeddings
            
        except KeyError as e:
            raise KeyError(f"News ID not found in embedding dictionary: {e}")



