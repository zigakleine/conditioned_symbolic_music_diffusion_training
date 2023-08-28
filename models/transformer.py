import math
import torch
from torch import nn, Tensor
import torch.nn.functional as F


class TransformerDDPM(nn.Module):

    def __init__(self, device):
        super().__init__()

        self.batch_size = 16
        self.seq_len = 8
        self.vocab_size = 64
        self.num_timesteps = 1000

        self.embed_size = 128

        self.num_heads = 8
        self.num_layers = 6

        self.num_mlp_layers = 2
        self.mlp_dims = 2048
        self.device = device

        self.token_embedding = nn.Linear(self.vocab_size, self.embed_size)
        self.position_embedding = nn.Embedding(self.seq_len, self.embed_size)
        self.timestep_embedding = nn.Embedding(self.num_timesteps, self.embed_size)

        self.layers = nn.ModuleList([EncoderLayer(self.embed_size, self.num_heads) for _ in range(self.num_layers)])

        # self.transformer_norm_1 = nn.LayerNorm(self.embed_size)
        # self.to_mlp_layer = nn.Linear(self.embed_size, self.mlp_dims)

        # self.mlp_layers = nn.ModuleList([EncoderLayer(self.embed_size, self.num_heads) for _ in range(self.num_mlp_layers)])

        self.transformer_norm_2 = nn.LayerNorm(self.embed_size)
        self.lm_head = nn.Linear(self.embed_size, self.vocab_size) # dense layer for output



    def forward(self, x, t):

        B, T, C = x.shape
        t1 = t[:, None].repeat(1, 8)

        tok_embedding = self.token_embedding(x)  # tok_embedding =  B, T, C
        pos_embedding = self.position_embedding(torch.arange(T, device=self.device))  # pos_embedding =  T, C
        timestep_embedding = self.timestep_embedding(t1)  # timestep_embedding = B, T, C
        x = tok_embedding + pos_embedding
        x += timestep_embedding

        for layer in self.layers:
            x = layer(x)

        # x = self.transformer_norm_1(x)
        # x = self.to_mlp_layer(x)
        #
        # for layer in self.mlp_layers:
        #     x = layer(x)

        x = self.transformer_norm_2(x)
        logits = self.lm_head(x)  # logits = B, T, vocab_size

        return logits


class EncoderLayer(nn.Module):

    def __init__(self, embed_size, num_heads, dim_feedforward=2048, dropout=0.1):

        super().__init__()

        self.self_attn = nn.MultiheadAttention(embed_size, num_heads, dropout=dropout)

        self.linear_1 = nn.Linear(embed_size, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(dim_feedforward, embed_size)

        self.norm_1 = nn.LayerNorm(embed_size)
        self.norm_2 = nn.LayerNorm(embed_size)
        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)

    def forward(self, x):

        shortcut = x
        x = self.norm_1(x)
        x = self.self_attn(x)
        x = x + shortcut

        shortcut_2 = x
        x = self.norm_2(x)
        x = self.linear_1(x)
        x = nn.GELU(x)
        x = self.linear_2(x)
        x = x + shortcut_2

        return x

# class MlpLayer(nn.Module):
#
#     def __init__(self):
#         pass
#
#     def forward(self, x):
#         pass



    # def apply(self, inputs, t, num_layers=6, num_heads=8, num_mlp_layers=2, mlp_dims=2048):
    #
    # batch_size, seq_len, data_channels = inputs.shape
    #
    # x = inputs
    # embed_channels = 128
    #
    # temb = PositionalEncoding(embed_channels, seq_len)
    # temb = temb[None, :, :]
    # assert temb.shape[1:] == (seq_len, embed_channels), temb.shape
    # x = nn.Dense(x, embed_channels)
    #
    # x = x + temb
    # for _ in range(num_layers):
    #   shortcut = x
    #   x = nn.LayerNorm(x)
    #   x = nn.SelfAttention(x, num_heads=num_heads)
    #   x = x + shortcut
    #   shortcut2 = x
    #   x = nn.LayerNorm(x)
    #   x = nn.Dense(x, mlp_dims)
    #   x = nn.gelu(x)
    #   x = nn.Dense(x, embed_channels)
    #   x = x + shortcut2
    #
    # x = nn.LayerNorm(x)
    # x = nn.Dense(x, mlp_dims)
    #
    # for _ in range(num_mlp_layers):
    #   scale, shift = DenseFiLM(t.squeeze(-1), 128, mlp_dims, sequence=True)
    #   x = DenseResBlock(x, mlp_dims, scale=scale, shift=shift)
    #
    # x = nn.LayerNorm(x)
    # x = nn.Dense(x, data_channels)
    # return x

class PositionalEncoding(nn.Module):

    def __init__(self, embed_channels, timesteps, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(timesteps).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_channels, 2) * (-math.log(10000.0) / embed_channels))
        pe = torch.zeros(timesteps, 1, embed_channels)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


if __name__ == "__main__":
    tr = TransformerDDPM()

    x = torch.ones(16, 8, 64)
    t = torch.randint(low=1, high=1000, size=(16,))

    timestep_embedding = nn.Embedding(1000, 64)

    t1 = t[:, None].repeat(1, 8)
    t2 = timestep_embedding(t1)

    print(t2 + x)
   # tr.forward(x, t)