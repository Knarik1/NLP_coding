import torch
import torch.nn as nn
from torch import Tensor
import torchtext

import TokenEmbedding, PositionalEncoding

class Seq2SeqTransformer(nn.Module):
    def __init__(self,
                 num_encoder_layers: int,
                 num_decoder_layers: int,
                 embed_size: int,
                 nhead: int,
                 src_vocab_size: int,
                 trg_vocab_size: int,
                 dim_feedforward: int = 512,
                 dropout_p: float = 0.1) -> None:
        super(Seq2SeqTransformer, self).__init__()

        self.transformer = nn.Transformer(d_model=embed_size,
                                          nhead=nhead,
                                          num_encoder_layers=num_encoder_layers,
                                          num_decoder_layers=num_decoder_layers,
                                          dim_feedforward=dim_feedforward,
                                          dropout=dropout_p)
        self.generator = nn.Linear(embed_size, trg_vocab_size)
        self.source_embedding = TokenEmbedding(src_vocab_size, embed_size)
        self.target_embedding = TokenEmbedding(trg_vocab_size, embed_size)
        self.positional_encoding = PositionalEncoding(embed_size, dropout_p)

    def forward(self,
                src: Tensor,
                trg: Tensor,
                src_mask: Tensor,
                trg_mask: Tensor,
                src_padding_mask: Tensor,
                trg_padding_mask: Tensor,
                memory_key_padding_mask: Tensor) -> Tensor:

        src_emb = self.positional_encoding(self.source_embedding(src))
        trg_emb = self.positional_encoding(self.target_embedding(trg))
        outs = self.transformer(src_emb, trg_emb, src_mask, trg_mask, None, src_padding_mask, trg_padding_mask, memory_key_padding_mask)
        outs = self.generator(outs)

        return outs

    def encode(self, src: Tensor, src_mask: Tensor) -> Tensor:
        return self.transformer.encoder(self.positional_encoding(self.source_embedding(src)), src_mask)

    def decode(self, trg: Tensor, memory: Tensor, trg_mask: Tensor) -> Tensor:
        return self.transformer.encoder(self.positional_encoding(self.target_embedding(trg)), memory, trg_mask)


def generate_square_subsequent_mask(size: int) -> Tensor:
    mask = (torch.triu(torch.ones((size, size))) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask


def create_mask(src, trg):
    pass


if __name__ == '__main__':
    """
     Used sources
        https://pytorch.org/tutorials/beginner/translation_transformer.html
        
    """
    print(torch.__version__)  # 1.9.0+cu102
    print(torchtext.__version__)  # 0.10.0
    sz = 10
    mask  = generate_square_subsequent_mask(10)
    print(mask)
