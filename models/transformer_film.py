import math
import torch
from torch import nn, Tensor
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer

#from torchinfo import summary
import numpy as np

class TransformerDDPM(nn.Module):

    def __init__(self, categories):
        super(TransformerDDPM, self).__init__()

        # self.batch_size = 16
        self.seq_len = 32
        self.vocab_size = 76
        self.num_timesteps = 1000

        self.embed_size = 512

        self.num_heads = 8
        self.num_layers = 12

        self.num_mlp_layers = 4
        self.mlp_dims = 2048
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.token_embedding = nn.Linear(self.vocab_size, self.embed_size)
        self.position_embedding = nn.Embedding(self.seq_len, self.embed_size)
        # self.timestep_embedding = nn.Embedding(self.num_timesteps, self.embed_size)

        self.layers = nn.ModuleList([EncoderLayer(self.embed_size, self.num_heads) for _ in range(self.num_layers)])

        self.transformer_norm_1 = nn.LayerNorm(self.embed_size)
        self.to_mlp_layers = nn.Linear(self.embed_size, self.mlp_dims)

        self.mlp_layers = nn.ModuleList([MlpLayer(self.embed_size, self.mlp_dims, categories) for _ in range(self.num_mlp_layers)])

        self.transformer_norm_2 = nn.LayerNorm(self.mlp_dims)
        self.lm_head = nn.Linear(self.mlp_dims, self.vocab_size) # dense layer for output

    def forward(self, x, t, genres, composers):

        B, T, C = x.shape
        # t1 = t[:, None].repeat(1, 8)

        tok_embedding = self.token_embedding(x)  # tok_embedding =  B, T, C
        pos_embedding = self.position_embedding(torch.arange(T, device=self.device))  # pos_embedding =  T, C
        # timestep_embedding = self.timestep_embedding(t1)  # timestep_embedding = B, T, C
        x = tok_embedding + pos_embedding
        # x += timestep_embedding

        for layer in self.layers:
            x = layer(x)

        x = self.transformer_norm_1(x)
        x = self.to_mlp_layers(x)

        for mlp_layer in self.mlp_layers:
            x = mlp_layer(x, t,  genres, composers)

        x = self.transformer_norm_2(x)
        logits = self.lm_head(x)  # logits = B, T, vocab_size

        return logits


class EncoderLayer(nn.Module):


    def __init__(self, embed_size, num_heads, dropout=0.1):

        super(EncoderLayer, self).__init__()

        dim_feedforward = embed_size * 4
        self.self_attn = nn.MultiheadAttention(embed_size, num_heads, dropout=dropout, batch_first=True)

        self.linear_1 = nn.Linear(embed_size, dim_feedforward)
        # self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(dim_feedforward, embed_size)

        self.norm_1 = nn.LayerNorm(embed_size)
        self.norm_2 = nn.LayerNorm(embed_size)
        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)
        self.dropout_3 = nn.Dropout(dropout)

        self.gelu = nn.GELU()

    def forward(self, x):

        shortcut = x
        x = self.norm_1(x)
        x, ws = self.self_attn(x, x, x, need_weights=False)
        x = self.dropout_1(x)
        x += shortcut

        shortcut_2 = x
        x = self.norm_2(x)
        x = self.linear_1(x)
        x = self.gelu(x)
        x = self.dropout_2(x)
        x = self.linear_2(x)
        x = self.dropout_3(x)
        x += shortcut_2

        return x


class MlpLayer(nn.Module):

    def __init__(self, timestep_embed_channels, mlp_dims, categories):
        super(MlpLayer, self).__init__()

        self.denseFiLM = DenseFiLM(timestep_embed_channels, mlp_dims, categories)
        self.denseResBlock = DenseResBlock(mlp_dims)

    def forward(self, x, t, genres, composers):

        scale, shift = self.denseFiLM(t, genres, composers)
        x = self.denseResBlock(x, scale, shift)
        return x


class DenseFiLM(nn.Module):

    def __init__(self, embed_channels, out_channels, categories, dropout=0.1):
        super(DenseFiLM, self).__init__()

        self.categories = categories
        self.num_genres = categories["genres"]
        self.num_composers = categories["composers"]

        self.embed_channels = embed_channels
        self.out_channels = out_channels
        self.linear_1 = nn.Linear(self.embed_channels,  self.embed_channels*4)
        self.linear_2 = nn.Linear(self.embed_channels*4,  self.embed_channels*4)

        if self.num_composers is not None:
            self.composers_emb = nn.Embedding(self.num_composers, self.embed_channels*4)

        if self.num_genres is not None:
            self.genres_emb = nn.Embedding(self.num_genres, self.embed_channels*4)

        self.scale = nn.Linear(self.embed_channels*4,  self.out_channels)
        self.shift = nn.Linear(self.embed_channels*4,  self.out_channels)

        self.silu = nn.SiLU()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def forward(self, t, genres, composers):


        t_embedding = self.positional_timestep_embedding(t, self.embed_channels)

        t_embedding = self.linear_1(t_embedding)
        t_embedding = self.silu(t_embedding)
        t_embedding = self.linear_2(t_embedding)

        if genres[0] != -1:
            t_embedding += self.genres_emb(genres)

        if composers[0] != -1:
            t_embedding += self.composers_emb(composers)

        scale_embedding = self.scale(t_embedding)
        shift_embedding = self.shift(t_embedding)

        return scale_embedding, shift_embedding

    def positional_timestep_embedding(self, timesteps, channels):
        # batch_size =
        # # channels = 128
        # assert timesteps.shape == (batch_size, 1)
        # channels.shape = ()
        noise = timesteps.squeeze(-1)
        assert len(noise.shape) == 1
        half_dim = channels // 2
        emb = math.log(10000) / float(half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=self.device) * -emb)
        emb = 5000 * noise[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
        if channels % 2 == 1:
            emb = torch.pad(emb, [[0, 0], [0, 1]])
        assert emb.shape == (noise.shape[0], channels)

        # print(emb)
        return emb


class DenseResBlock(nn.Module):

    def __init__(self, out_channels, dropout=0.1):
        super(DenseResBlock, self).__init__()

        self.linear_1 = nn.Linear(out_channels, out_channels)
        self.linear_2 = nn.Linear(out_channels, out_channels)

        self.norm_1 = nn.LayerNorm(out_channels)
        self.norm_2 = nn.LayerNorm(out_channels)

        self.silu = nn.SiLU()

    def forward(self, x, scale, shift):
        B, T, C = x.shape

        scale = scale.view(B, 1, C)  # Reshape scale to (b, 1, c)
        scale = scale.expand(B, T, C)

        shift = shift.view(B, 1, C)  # Reshape shift to (b, 1, c)
        shift = shift.expand(B, T, C)


        shortcut = x
        x = self.norm_1(x)
        x = scale * x + shift
        x = self.silu(x)
        x = self.linear_1(x)

        x = self.norm_2(x)
        x = scale * x + shift
        x = self.silu(x)
        x = self.linear_2(x)

        return x + shortcut

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

if __name__ == "__main__":
    # pass
    categories = {'genres': 13, 'composers': 292}

    seq_len = 32
    vocab_size = 76
    num_timesteps = 1000

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TransformerDDPM(categories).to(device)
    print(model)
    print(count_parameters(model))
    #
    #
    # summary(model, [(64, 32, 76), (64, 1), (64, 1), (64, 1)], batch_dim=0)
    # for p in model.parameters():
    #     print(p)
    #
    # x_ = torch.ones(64, 32, 42)
    # t_ = torch.randint(low=1, high=1000, size=(64, 1))
    # out = tr(x_, t_)



    # for name, param in model.named_parameters():
    #     if param.requires_grad:
    #         print (name, param.data)


    # t1 = t[:, None].repeat(1, 8)
    # t2 = timestep_embedding(t1)
    #
    # print(t2 + x)
    # tr.forward(x, t)


    # b, c
    # x_ = torch.ones(16, 8, 64)
    # emb = torch.randn(16, 64)
    #
    # b, t, c = x_.shape
    # emb111 = emb
    # emb = emb.view(b, 1, c)  # Reshape emb to (b, 1, c)
    # emb = emb.expand(b, t, c)
    # print(emb)
    # print(emb.shape)