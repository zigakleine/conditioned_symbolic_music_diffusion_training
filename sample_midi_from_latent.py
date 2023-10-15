import os
import numpy as np
from diffusion_main import Diffusion, inverse_data_transform
import torch
import pickle
from models.transformer_film import TransformerDDPM
from singletrack_VAE import singletrack_vae, db_processing
import uuid


current_dir = os.getcwd()

file_to_sample_abs_path = "./29000_epoch_batch.pkl"
sampled_latents = pickle.load(open(file_to_sample_abs_path, "rb"))

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

z = sampled_latents[0]

song_data_ = vae.decode_sequence(z, total_steps, temperature)

midi = db_proc.midi_from_song(song_data_)
midi.save("./overfit-1510.mid")
