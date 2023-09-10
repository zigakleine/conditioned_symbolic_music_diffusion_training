import os
import numpy as np
from diffusion_main import Diffusion, inverse_data_transform
import torch
import pickle
from models.transformer_film import TransformerDDPM
from multitrack_VAE import multitrack_vae, db_processing
import uuid




current_dir = os.getcwd()

file_to_sample_abs_path = "./140_epoch_batch.pkl"
sampled_latents = pickle.load(open(file_to_sample_abs_path, "rb"))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_rel_path = "multitrack_vae_model/model_fb256.ckpt"
batch_size = 32
total_steps = 512
temperature = 0.2
nesmdb_shared_library_rel_path = "ext_nseq_nesmdb_lib.so"
db_type = "nesmdb"
model_path = os.path.join(current_dir, model_rel_path)
nesmdb_shared_library_path = os.path.join(current_dir, nesmdb_shared_library_rel_path)


vae = multitrack_vae(model_path, batch_size)
db_proc = db_processing(nesmdb_shared_library_path, db_type)

sampled_latents = sampled_latents.reshape(sampled_latents.shape[0] * sampled_latents.shape[1],
                                              sampled_latents.shape[-1])

decoded_song = vae.decode_sequence(sampled_latents, total_steps, temperature)

generated_midi = db_proc.midi_from_song(decoded_song)
generated_midi.save("./song_from_latent.mid")
