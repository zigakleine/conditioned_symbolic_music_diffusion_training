import pickle
import numpy as np
import torch.nn as nn
import torch

original_song_path = "29000_epoch_batch.pkl"
original_song = pickle.load(open(original_song_path, "rb"))
original_song = original_song[0]

training_song_path = "./mario_encoded.pkl"
training_song = pickle.load(open(training_song_path, "rb"))
training_song = training_song[0]

# print(np.sum(training_song - original_song))
mse = nn.MSELoss()
loss = mse(torch.tensor(training_song), torch.tensor(original_song))
loss = loss.item()

print(loss)