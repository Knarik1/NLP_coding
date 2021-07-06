import torch
import torch.nn as nn


class SelfAttention(nn.Module):
    def __init__(self, embed_dim: int) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        # Shape (embed_dim, embed_dim)
        self.W_Q = nn.Parameter(torch.Tensor(embed_dim, embed_dim))
        self.W_K = nn.Parameter(torch.Tensor(embed_dim, embed_dim))
        self.W_V = nn.Parameter(torch.Tensor(embed_dim, embed_dim))

        nn.init.norm_(self.W_Q)
        nn.init.norm_(self.W_K)
        nn.init.norm_(self.W_V)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Shape x is (bs, num_tokens, vocab_size)
        # Shapes Q, K, V are (bs, num_tokens, embed_dim)
        Q = x @ self.W_Q
        K = x @ self.W_K
        V = x @ self.W_V

        # Shape (bs, num_tokens, num_tokens)
        Z_1 = torch.bmm(Q, K.transpose(1,2))
        Z_2 = Z_1 / self.embed_dim ** 0.5
        Z_3 = nn.Softmax(Z_2, dim=-1)
        # Shape (bs, num_tokens, embed_dim)
        Z = torch.bmm(Z_3, V)

        return Z

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, nheads: int, embed_dim: int) -> None:
        super(MultiHeadSelfAttention, self).__init__()

        assert embed_dim % nheads == 0

        self.nheads = nheads
        self.embed_dim = embed_dim
        self.heads = nn.ModuleList([
            SelfAttention(embed_dim // nheads) for i in range(nheads)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        Z = [head(x) for head in self.heads]
        multiHead = torch.cat(Z, dim=1)

        return multiHead


class TransformerEncoderLayer(nn.Module):
    def __init__(self, embed_dim: int, nheads: int, ff_dim: int) -> None:
        super(TransformerEncoderLayer, self).__init__()
        self.embed_dim = embed_dim
        self.nheads = nheads
        self.output_dim = output_dim

        self.multiHead = MultiHeadSelfAttention(nheads, embed_dim)
        self.layerNorm_selfAttention = nn.LayerNorm(embed_dim)
        self.layerNorm_ff = nn.LayerNorm(embed_dim)
        self.ff1 = nn.Linear(embed_dim, ff_dim)
        self.ff2 = nn.Linear(ff_dim, output_dim)
        self.activation = nn.Relu()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Shape x (bs, nu_tokens, embed_dim)
        # Shape (bs, nu_tokens, embed_dim)
        Z = self.multiHead(x_embed)
        x_normed = self.layerNorm_selfAttention(x_embed + Z)
        output = self.activation(self.ff_1(x_normed))
        output = self.ff_2(output)
        output_normed = self.layerNorm_ff(x_normed + output)

        return output_normed


class TransformerEncoder(nn.Module):
    def __init__(self, embed_dim: int, nheads: int, ff_dim: int, n_layers: int) -> None:
        super(TransformerEncoder, self).__init__()
        self.embed_dim = embed_dim
        self.nheads = nheads
        self.ff_dim = ff_dim
        self.n_layers = n_layers

        self.layers = nn.ModuleList([TransformerEncoderLayer(embed_dim, nheads, ff_dim) for _ in range(n_layers)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input = x
        for layer in self.layers():
            output = layer(input)
            input = output

        return input


if __name__ == '__main__':
    embed_dim = 2
    print(torch.Tensor(embed_dim, embed_dim))
