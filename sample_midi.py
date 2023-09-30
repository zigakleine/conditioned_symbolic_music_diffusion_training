import os
import numpy as np
from diffusion_main import Diffusion, inverse_data_transform
import torch
import pickle
from models.transformer_film_emotion import TransformerDDPME
from singletrack_VAE import singletrack_vae, db_processing
import uuid
import random

def sample_midi():

    random_emotions = [random.randint(0, 3) for _ in range(num_samples_to_generate)]
    random_emotions = torch.tensor(random_emotions).to(device)
    sampled_latents = diffusion.sample(model, num_samples_to_generate, random_emotions, cfg_scale=3)
    batch_transformed = inverse_data_transform(torch.Tensor.cpu(sampled_latents), -14., 14.)
    batch_split = np.split(batch_transformed[0], 4, axis=1)
    batch_ = np.vstack(batch_split)

    decoded_song = vae.decode_sequence(batch_, total_steps, temperature)
    decoded_song_survey = decoded_song[:15]
    generated_midi_survey = db_proc.midi_from_song(decoded_song_survey)
    generated_midi_eval = db_proc.midi_from_song(decoded_song)
    return generated_midi_survey, generated_midi_eval


current_dir = os.getcwd()
num_samples_to_generate = 1
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 64

categories = {"emotions": 4}
model = TransformerDDPME(categories).to(device)

diffusion = Diffusion(noise_steps=model.num_timesteps, batch_size=batch_size, vocab_size=model.vocab_size,
                      time_steps=model.seq_len)
checkpoint = torch.load("./min_checkpoint.pth.tar", map_location=device)
model.load_state_dict(checkpoint["state_dict"])
print("epoch:", checkpoint["epoch"])

temperature = 0.0002
total_steps = 32
model_rel_path = "cat-mel_2bar_big.tar"
model_path = os.path.join(current_dir, model_rel_path)
db_type = "nesmdb_singletrack"
nesmdb_shared_library_rel_path = "ext_nseq_nesmdb_single_lib.so"

model_path = os.path.join(current_dir, model_rel_path)
nesmdb_shared_library_path = os.path.join(current_dir, nesmdb_shared_library_rel_path)

db_proc = db_processing(nesmdb_shared_library_path, db_type)
vae = singletrack_vae(model_path, batch_size)

run_folder_name = "nes"
survey_samples_folder_name = "samples_survey_" + run_folder_name
eval_samples_folder_name = "samples_eval_" + run_folder_name

survey_samples_dir = os.path.join(current_dir, survey_samples_folder_name)
eval_samples_dir = os.path.join(current_dir, eval_samples_folder_name)

if not os.path.exists(survey_samples_dir):
    os.mkdir(survey_samples_dir)

if not os.path.exists(eval_samples_dir):
    os.mkdir(eval_samples_dir)

for i in range(100):
    uuid_string = str(uuid.uuid4())
    print("iteration:", i)
    midi_output_path_survey = os.path.join(survey_samples_dir, uuid_string + ".mid")
    midi_output_path_eval = os.path.join(eval_samples_dir, "eval_" + str(i) + ".mid")

    generated_midi_survey, generated_midi_eval = sample_midi()
    generated_midi_survey.save(midi_output_path_survey)
    generated_midi_eval.save(midi_output_path_eval)
