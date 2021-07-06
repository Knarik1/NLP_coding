import math

import torch
import torch.nn as nn
import torchtext
from torch import Tensor

class TokenEmbedding(nn.Module):
    def __init__(self,
                 vocab_size: int,
                 embedding_dim: int) -> None:
        super(TokenEmbedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.embedding_dim = embedding_dim

    def forward(self, tokens: Tensor) -> Tensor:
        output = self.embedding(tokens.long()) * math.sqrt(self.embedding_dim)

        return output


if __name__ == '__main__':
    """
     Used sources
        https://pytorch.org/tutorials/beginner/translation_transformer.html
        
    """
    print(torch.__version__)  # 1.9.0+cu102
    print(torchtext.__version__)  # 0.10.0

    vocab_size = 500
    embedding_dim = 300
    maxlen = 100

    token = TokenEmbedding(vocab_size, embedding_dim)
    tokens = torch.arange(maxlen)
    output = token(tokens)
    print(output.shape)
