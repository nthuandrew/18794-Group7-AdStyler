import torch
import torch.nn as nn

class SimpleMetadataEncoder(nn.Module):
    def __init__(self, metadata_vocab_size: int, embed_dim: int):
        super().__init__()
        self.emb = nn.Embedding(metadata_vocab_size, embed_dim)

    def forward(self, metadata_ids: torch.LongTensor) -> torch.Tensor:
        # metadata_ids: (batch,)
        return self.emb(metadata_ids)  # returns (batch, embed_dim)