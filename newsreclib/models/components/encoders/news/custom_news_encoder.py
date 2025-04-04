from typing import List
import torch
from torch import nn

class CustomNewsEncoder(nn.Module):
    def __init__(self, custom_embedding_path: str):
        super().__init__()
        self.custom_embedding_path = custom_embedding_path
        embeddings_dict = torch.load(custom_embedding_path)
        print(f"Custom embeddings loaded with {len(embeddings_dict)} entries.")
        embeddings_tensor = torch.stack([torch.tensor(e) for e in embeddings_dict.values()])
        self.embedding = nn.Embedding.from_pretrained(embeddings_tensor, freeze=True)
        self.news_id_to_index = {news_id: idx for idx, news_id in enumerate(embeddings_dict.keys())}

    def forward(self, news_ids: List[str]):
        indices = torch.tensor([self.news_id_to_index[news_id] for news_id in news_ids], device=self.embedding.weight.device)
        embeddings = self.embedding(indices)

        # Add explicit assert or debug statement here:
        assert embeddings.shape[-1] == 300, f"Embedding dim mismatch: {embeddings.shape}"
        return embeddings



