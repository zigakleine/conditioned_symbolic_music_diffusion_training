import pickle
import numpy as np
import torch.nn as nn
import torch
import os
from singletrack_VAE import singletrack_vae, db_processing

original_song_path = "./30000_epoch_batch.pkl"
original_song = pickle.load(open(original_song_path, "rb"))
original_song = original_song[0]

training_song_path = "./mario_encoded.pkl"
training_song = pickle.load(open(training_song_path, "rb"))
training_song = training_song[0]

# print(np.sum(training_song - original_song))
mse = nn.MSELoss()
loss = mse(torch.tensor(training_song), torch.tensor(original_song))
loss = loss.item()

print(f"latents-mse-loss-{loss}")

current_dir = os.getcwd()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
nesmdb_shared_library_rel_path = "ext_nseq_nesmdb_single_lib.so"

batch_size = 64
temperature = 0.0002
total_steps = 32

current_dir = os.getcwd()
model_rel_path = "cat-mel_2bar_big.tar"
model_path = os.path.join(current_dir, model_rel_path)
db_type = "nesmdb_singletrack"
nesmdb_shared_library_rel_path = "ext_nseq_nesmdb_single_lib.so"

model_path = os.path.join(current_dir, model_rel_path)
nesmdb_shared_library_path = os.path.join(current_dir, nesmdb_shared_library_rel_path)

db_proc = db_processing(nesmdb_shared_library_path, db_type)
vae = singletrack_vae(model_path, batch_size)

mario_path = "./mario_unaltered.mid"
mario_song = db_proc.song_from_midi_nesmdb(mario_path, 0, True)
mario_song = mario_song[:, :16*32]


song_data_new = vae.decode_sequence(original_song, total_steps, temperature)
song_data_tr = vae.decode_sequence(training_song, total_steps, temperature)

compare_songs = (song_data_new == song_data_tr)
true_count = np.count_nonzero(compare_songs)
total_elements = compare_songs.size

compare_songs_mario = (song_data_new == mario_song)
true_count_m = np.count_nonzero(compare_songs_mario)
total_elements_m = compare_songs_mario.size

accuracy = true_count / total_elements
accuracy_m = true_count_m / total_elements_m

print(f"accuracy-traininglatent-generated-{accuracy}")
print(f"accuracy-origmario-generated-{accuracy_m}")
