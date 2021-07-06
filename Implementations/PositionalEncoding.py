import math

import torch
from torch import Tensor
import torch.nn as nn
import torchtext


class PositionalEncoding(nn.Module):
    """
     A positional encoding module for the token embedding to introduce a notion of word order.
     PositionalEncoding module injects some information about the relative or absolute position
     of the tokens in the sequence. The positional encodings have the same dimension as the
     embeddings so that the two can be summed. Here, we use sine and cosine functions of
     different frequencies.

     Args:
            embedding_dim: embedding dimention
            dropout_p: probability of an element to be zeroed. Default: 0.5
            maxlen: max length of a token vector, vocabulary size
    """
    def __init__(self,
                 embedding_dim: int,
                 dropout_p: float = 0.1,
                 maxlen: int = 5000) -> None:

        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout_p)

        # shape (maxlen, embedding_dim)
        pos_embedding = torch.zeros(maxlen, embedding_dim)

        # shape (maxlen, 1)
        pos = torch.arange(0, maxlen).reshape(maxlen, 1)
        # shape (embedding_dim/2)
        den = torch.exp(- torch.arange(0, embedding_dim, 2)* math.log(10000) / embedding_dim)

        pos_embedding[:, 0::2] = torch.sin(pos * den)
        pos_embedding[:, 1::2] = torch.cos(pos * den)

        # shape (maxlen, 1, embedding_dim)
        pos_embedding = pos_embedding.unsqueeze(-2)

        self.register_buffer('pos_embedding', pos_embedding)

    def forward(self, token_embedding: Tensor) -> Tensor:
        # shape (token_embedding, 1, token_embedding) ???
        output = token_embedding + self.pos_embedding[:token_embedding.size(0), :]
        output = self.dropout(output)

        return output

if __name__ == '__main__':
    """
     Used sources
        https://pytorch.org/tutorials/beginner/translation_transformer.html
        
    """
    print(torch.__version__)  # 1.9.0+cu102
    print(torchtext.__version__)  # 0.10.0

    embedding_dim = 300
    maxlen = 1700
    dropout_p = 0.2

    token_embedding = torch.rand(embedding_dim)
    pos_encoder = PositionalEncoding(embedding_dim, dropout_p, maxlen)
    output = pos_encoder(token_embedding)
    print(output.shape)
    print(output)

