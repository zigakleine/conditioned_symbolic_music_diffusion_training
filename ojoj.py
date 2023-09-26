import math
import torch
import numpy as np
import torch
import pickle

#
# device = "cpu"
# batch_size = 32
# channels = 128
# noise_steps = 1000
#
# timesteps_2 = np.random.choice(noise_steps, batch_size)
# timesteps = torch.randint(low=1, high=noise_steps, size=(batch_size, 1))
#
#
# assert timesteps.shape == (batch_size, 1)
#
# noise = timesteps.squeeze(-1)
# assert len(noise.shape) == 1
# half_dim = channels // 2
# emb = math.log(10000) / float(half_dim - 1)
# emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
# emb = 5000 * noise[:, None] * emb[None, :]
# emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
# if channels % 2 == 1:
#     emb = torch.pad(emb, [[0, 0], [0, 1]])
# assert emb.shape == (noise.shape[0], channels)




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

# enc_seq_abs_path = "./nesmdb_encoded/282_RoboWarrior/1*+0*p1-p2-tr-no.pkl"
# enc_seq_abs_path = "./0f57cce4d23ab451a13e7826791d922a_enc.pkl"
# enc_seq_abs_path = "./3*-3*p1-p2-tr-no.pkl"
enc_seq_abs_path = "./12*+0*p1-p2-tr-no.pkl"
enc_seq = pickle.load(open(enc_seq_abs_path, "rb"))
enc_seq = enc_seq[0]

enc_seq_tracks = np.split(enc_seq, 4, axis=0)
enc_seq_hstacked = np.hstack(enc_seq_tracks)

enc_seq_tracks_ = np.split(enc_seq_hstacked, 4, axis=1)
enc_seq_decoded = np.vstack(enc_seq_tracks_)

print(enc_seq == enc_seq_decoded)
print(enc_seq_decoded)