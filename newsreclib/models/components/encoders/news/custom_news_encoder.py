from typing import List
import torch
import logging
from torch import nn
from pathlib import Path
import urllib.request

logger = logging.getLogger(__name__)

class CustomNewsEncoder(nn.Module):
    """Encoder that uses pre-computed news embeddings.
    
    This encoder loads pre-computed news embeddings (e.g., from LLMs) directly,
    rather than computing them from word embeddings.
    """
    def __init__(self, custom_embedding_path: str):
        super().__init__()
        self.custom_embedding_path = custom_embedding_path
        
        # If path is URL, download first
        if custom_embedding_path.startswith(('http://', 'https://')):
            local_path = Path('/tmp/custom_embeddings.pt')
            logger.info(f"Downloading embeddings from {custom_embedding_path}")
            urllib.request.urlretrieve(custom_embedding_path, local_path)
            custom_embedding_path = str(local_path)
        
        try:
            embeddings_dict = torch.load(custom_embedding_path)
            logger.info(f"Loaded {len(embeddings_dict)} pre-computed news embeddings")
            
            # Convert to embedding layer
            embeddings_tensor = torch.stack([torch.tensor(e) for e in embeddings_dict.values()])
            self.embedding = nn.Embedding.from_pretrained(embeddings_tensor, freeze=True)
            self.news_id_to_index = {news_id: idx for idx, news_id in enumerate(embeddings_dict.keys())}
            
            embed_dim = embeddings_tensor.shape[-1]
            logger.info(f"Embedding dimension: {embed_dim}")
            
        except Exception as e:
            raise RuntimeError(f"Failed to load embeddings from {custom_embedding_path}. Error: {str(e)}")

    def forward(self, news_ids: List[str]):
        try:
            indices = torch.tensor([self.news_id_to_index[news_id] for news_id in news_ids], 
                                 device=self.embedding.weight.device)
            return self.embedding(indices)
        except KeyError as e:
            raise KeyError(f"News ID not found in embedding dictionary: {e}")



