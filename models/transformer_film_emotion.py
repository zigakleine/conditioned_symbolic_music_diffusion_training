import math
import torch
from torch import nn, Tensor
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer

#from torchinfo import summary
import numpy as np

class TransformerDDPME(nn.Module):

    def __init__(self, categories):
        super(TransformerDDPME, self).__init__()

        self.seq_len = 16
        self.vocab_size = 2048

        self.num_timesteps = 1000

        self.embed_size = 2280
        # self.embed_size = 2048

        self.num_heads = 12
        # self.num_heads = 8
        self.num_layers = 5

        self.num_mlp_layers = 3
        # self.num_mlp_layers = 4
        self.mlp_dims = 2048
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.token_embedding = nn.Linear(self.vocab_size, self.embed_size)
        self.position_embedding = nn.Embedding(self.seq_len, self.embed_size)

        self.layers = nn.ModuleList([EncoderLayer(self.embed_size, self.num_heads) for _ in range(self.num_layers)])

        self.transformer_norm_1 = nn.LayerNorm(self.embed_size)
        self.to_mlp_layers = nn.Linear(self.embed_size, self.mlp_dims)

        self.mlp_layers = nn.ModuleList([MlpLayer(self.embed_size, self.mlp_dims, categories) for _ in range(self.num_mlp_layers)])

        self.transformer_norm_2 = nn.LayerNorm(self.mlp_dims)
        self.lm_head = nn.Linear(self.mlp_dims, self.vocab_size) # dense layer for output

    def transformer_timestep_embedding(self, timesteps, channels):
        noise = timesteps
        assert len(noise.shape) == 1
        half_dim = channels // 2
        emb = math.log(10000) / float(half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=self.device) * -emb)
        emb = noise[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
        if channels % 2 == 1:
            emb = torch.pad(emb, [[0, 0], [0, 1]])
        assert emb.shape == (noise.shape[0], channels)
        return emb

    def forward(self, x, t, emotions):

        B, T, C = x.shape


        tok_embedding = self.token_embedding(x)  # tok_embedding =  B, T, C
        pos_embedding = self.position_embedding(torch.arange(T, device=self.device))   # pos_embedding =  T, C
        # pos_embedding_ = self.transformer_timestep_embedding(torch.arange(T, device=self.device), self.embed_size)  # pos_embedding =  T, C
        # print(pos_embedding.requires_grad)

        x = tok_embedding + pos_embedding

        for layer in self.layers:
            x = layer(x)

        x = self.transformer_norm_1(x)
        x = self.to_mlp_layers(x)

        for mlp_layer in self.mlp_layers:
            x = mlp_layer(x, t, emotions)

        x = self.transformer_norm_2(x)
        logits = self.lm_head(x)  # logits = B, T, vocab_size

        return logits


class EncoderLayer(nn.Module):


    def __init__(self, embed_size, num_heads, dropout=0.1):

        super(EncoderLayer, self).__init__()

        dim_feedforward = embed_size * 4
        self.self_attn = nn.MultiheadAttention(embed_size, num_heads, dropout=dropout, batch_first=True)

        self.linear_1 = nn.Linear(embed_size, dim_feedforward)
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

    def forward(self, x, t, emotions):

        scale, shift = self.denseFiLM(t, emotions)
        x = self.denseResBlock(x, scale, shift)
        return x


class DenseFiLM(nn.Module):

    def __init__(self, embed_channels, out_channels, categories, dropout=0.1):
        super(DenseFiLM, self).__init__()

        self.categories = categories
        self.num_emotions = categories["emotions"]

        self.embed_channels = embed_channels
        self.embed_chanels_mul = self.embed_channels * 2
        self.out_channels = out_channels
        self.linear_1 = nn.Linear(self.embed_channels,  self.embed_chanels_mul)
        self.linear_2 = nn.Linear(self.embed_chanels_mul,  self.embed_chanels_mul)

        if self.num_emotions > 0:
            self.emotions_emb = nn.Embedding(self.num_emotions, self.embed_chanels_mul)

        self.scale = nn.Linear(self.embed_chanels_mul,  self.out_channels)
        self.shift = nn.Linear(self.embed_chanels_mul,  self.out_channels)

        self.silu = nn.SiLU()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def forward(self, t, emotions):

        t_embedding = self.positional_timestep_embedding(t, self.embed_channels)
        # print(t_embedding.requires_grad)
        t_embedding = self.linear_1(t_embedding)
        t_embedding = self.silu(t_embedding)
        t_embedding = self.linear_2(t_embedding)

        if emotions is not None:
            t_embedding += self.emotions_emb(emotions)

        scale_embedding = self.scale(t_embedding)
        shift_embedding = self.shift(t_embedding)

        return scale_embedding, shift_embedding

    def positional_timestep_embedding(self, timesteps, channels):
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

        return emb


class DenseResBlock(nn.Module):

    def __init__(self, out_channels, dropout=0.1):
        super(DenseResBlock, self).__init__()

        self.linear_1 = nn.Linear(out_channels, out_channels)
        self.linear_2 = nn.Linear(out_channels, out_channels)

        self.norm_1 = nn.LayerNorm(out_channels)
        self.norm_2 = nn.LayerNorm(out_channels)

        # self.dropout_1 = nn.Dropout(dropout)
        # self.dropout_2 = nn.Dropout(dropout)

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
        # x = self.dropout_1(x)

        x = self.norm_2(x)
        x = scale * x + shift
        x = self.silu(x)
        x = self.linear_2(x)
        # x = self.dropout_2(x)

        return x + shortcut

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

if __name__ == "__main__":
    # pass
    categories = {'emotions': 4}
    seq_len = 16
    vocab_size = 2048
    num_timesteps = 1000

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = TransformerDDPME(categories).to(device)
    print(model)
    print(count_parameters(model))

    # x_ = torch.ones(64, seq_len, vocab_size)
    # t_ = torch.randint(low=1, high=1000, size=(64, 1), dtype=torch.int64)
    # emotions = torch.ones(size=(64,), dtype=torch.int64)
    # # emotions = None
    # out = model(x_, t_, emotions)

    #
    #
    # summary(model, [(64, 32, 76), (64, 1), (64, 1), (64, 1)], batch_dim=0)
    # for p in model.parameters():
    #     print(p)
    #
    # x_ = torch.ones(64, seq_len, vocab_size)
    # t_ = torch.randint(low=1, high=1000, size=(64, 1))
    # genres = torch.tensor([-1], dtype=torch.int64)
    # composers = torch.tensor([-1], dtype=torch.int64)
    #
    # out = model(x_, t_, genres, composers)



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