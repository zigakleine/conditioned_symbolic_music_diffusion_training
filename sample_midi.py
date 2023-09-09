import os
import numpy as np
from diffusion_main import Diffusion, inverse_data_transform
import torch
import pickle
from models.transformer_film import TransformerDDPM
from multitrack_VAE import multitrack_vae, db_processing
import uuid

def sample_midi():
    genres = torch.tensor(np.random.choice(genres_num, 1), dtype=torch.int64)
    composers = torch.tensor(np.random.choice(composers_num, 1), dtype=torch.int64)

    # genres = torch.tensor([-1], dtype=torch.int64)
    # composers = torch.tensor([-1], dtype=torch.int64)

    sampled_latents = diffusion.sample(model, num_samples_to_generate, genres, composers, cfg_scale=0.5)
    batch_transformed = inverse_data_transform(torch.Tensor.cpu(sampled_latents), fb256_slices, min_max["min"],
                                               min_max["max"])

    batch_transformed = batch_transformed.reshape(batch_transformed.shape[0] * batch_transformed.shape[1],
                                                  batch_transformed.shape[-1])
    decoded_song = vae.decode_sequence(batch_transformed, total_steps, temperature)
    decoded_song_1 = decoded_song[:15]
    generated_midi_1 = db_proc.midi_from_song(decoded_song_1)
    generated_midi_2 = db_proc.midi_from_song(decoded_song)
    return generated_midi_1, generated_midi_2

current_dir = os.getcwd()

num_samples_to_generate = 1

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

categories_path = "./db_metadata/nesmdb/nesmdb_categories.pkl"
categories_indices = pickle.load(open(categories_path, "rb"))
genres_num = len(categories_indices["genres"].keys())
composers_num = len(categories_indices["composers"].keys())
categories = {"genres": genres_num, "composers": composers_num}
model = TransformerDDPM(categories).to(device)

diffusion = Diffusion()
# model_run_name = "ddpm_lakh_nesmdb"
model_run_name = "ddpm_nesmdb"
# existing_model_abs_path = os.path.join(current_dir, "checkpoints", model_run_name, "min_checkpoint_lakh_nesmdb.pth.tar")
existing_model_abs_path = os.path.join(current_dir, "checkpoints", model_run_name, "min_checkpoint_213_nesmdb.pth.tar")
checkpoint = torch.load(existing_model_abs_path, map_location=device)
model.load_state_dict(checkpoint["state_dict"])
print("epoch:", checkpoint["epoch"])

slice_ckpt = "./pkl_info/fb256_slices_76.pkl"
fb256_slices = pickle.load(open(slice_ckpt, "rb"))
min_max_ckpt_path = "./pkl_info/nesmdb_min_max.pkl"
min_max = pickle.load(open(min_max_ckpt_path, "rb"))



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


# generated_samples_folder_name = "lakh_nesmdb_diffusion_samples"
generated_samples_folder_name = "nesmdb_diffusion_samples"
# generated_samples_folder_name_eval = "lakh_nesmdb_diffusion_samples_eval"
generated_samples_folder_name_eval = "nesmdb_diffusion_samples_eval"
samples_out_dir = os.path.join(current_dir, generated_samples_folder_name)
samples_out_dir_eval = os.path.join(current_dir, generated_samples_folder_name_eval)

if not os.path.exists(samples_out_dir):
    os.mkdir(samples_out_dir)

if not os.path.exists(samples_out_dir_eval):
    os.mkdir(samples_out_dir_eval)

for i in range(100):
    uuid_string = str(uuid.uuid4())
    print(uuid_string)
    midi_output_path = os.path.join(samples_out_dir, uuid_string + ".mid")
    # midi_output_path_eval = os.path.join(samples_out_dir_eval, "eval_lakh_nesmdb_" + str(i) + ".mid")
    midi_output_path_eval = os.path.join(samples_out_dir_eval, "eval_nesmdb_" + str(i) + ".mid")
    generated_midi_1, generated_midi_2 = sample_midi()
    generated_midi_1.save(midi_output_path)
    generated_midi_2.save(midi_output_path_eval)