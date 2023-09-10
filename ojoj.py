import math
import torch
import numpy as np
import torch

device = "cpu"
batch_size = 32
channels = 128
noise_steps = 1000

timesteps_2 = np.random.choice(noise_steps, batch_size)
timesteps = torch.randint(low=1, high=noise_steps, size=(batch_size, 1))


assert timesteps.shape == (batch_size, 1)

noise = timesteps.squeeze(-1)
assert len(noise.shape) == 1
half_dim = channels // 2
emb = math.log(10000) / float(half_dim - 1)
emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
emb = 5000 * noise[:, None] * emb[None, :]
emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
if channels % 2 == 1:
    emb = torch.pad(emb, [[0, 0], [0, 1]])
assert emb.shape == (noise.shape[0], channels)




# a = np.zeros((2,3,3), dtype=np.int64)
# b = np.zeros((3,3), dtype=np.int64)
#
# for i in range(3):
#     for j in range(3):
#         b[i, j] = j
#
# c = a + b
#
# print(c)