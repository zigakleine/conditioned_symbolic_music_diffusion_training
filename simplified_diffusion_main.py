from diffusion_main import Diffusion
import tqdm
import logging
import os
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
from torchvision.transforms import functional as F
from PIL import Image
import pickle

def normalize_dataset(batch, data_min, data_max, std_dev_masks):
    """Normalize dataset to range [-1, 1]."""
    batch = (batch - data_min) / (data_max - data_min)
    batch = 2. * batch - 1.
    # # print("batch-mean-", batch.mean(axis=(0, 1)))
    #
    # enc_tracks = np.split(batch, 4, axis=0)
    # enc_tracks_reduced = []
    # for enc_track, std_dev_mask in zip(enc_tracks, std_dev_masks):
    #
    #     enc_track_reduced = enc_track[:, std_dev_mask]
    #     enc_tracks_reduced.append(enc_track_reduced)
    #
    # enc_tracks_reduced = np.vstack(enc_tracks_reduced)
    enc_tracks_reduced = batch
    return enc_tracks_reduced

def inverse_data_transform(batch, data_min, data_max, std_dev_masks):

    batch = (batch + 1.) / 2.
    batch = (data_max - data_min) * batch + data_min
    batch = batch.numpy()
    batch_ = []
    for enc_tracks in batch:

        enc_tracks_split = np.split(enc_tracks, 4, axis=1)
        enc_tracks_reconstructed = enc_tracks_split
        # enc_tracks_reconstructed = []
        # for enc_track, std_devs_mask in zip(enc_tracks_split, std_dev_masks):
        #     enc_track_reconstructed = np.random.randn(*enc_track.shape[:-1], 512)
        #     enc_track_reconstructed[..., std_devs_mask] = enc_track
        #     enc_tracks_reconstructed.append(enc_track_reconstructed)
            # transformed = np.random.randn(*batch.shape[:-1], out_channels)
            # transformed[..., slices] = batch
            # batch = transformed

        enc_tracks_reconstructed = np.vstack(enc_tracks_reconstructed)
        batch_.append(enc_tracks_reconstructed)

    return np.array(batch_)

def setup_logging(run_name, current_dir):

    os.makedirs(os.path.join(current_dir, "results"), exist_ok=True)
    os.makedirs(os.path.join(current_dir, "results", run_name), exist_ok=True)
    os.makedirs(os.path.join(current_dir, "results", run_name, "generated"), exist_ok=True)
    os.makedirs(os.path.join(current_dir, "results", run_name, "graphs"), exist_ok=True)


dmin = -5.
dmax = 5.
epochs_num = 5000
lr = 1.81e-5
batch_size = 1
current_dir = os.getcwd()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
training_data_type = "song"
run_name = "song_overfit_test_9"
# run_name = "img_overfit_test_1"

categories = {"emotions": 4}

if training_data_type == "img":
    from models.transformer_film_img import TransformerDDPME
elif training_data_type == "song":
    from models.transformer_film_emotion import TransformerDDPME

model = TransformerDDPME(categories).to(device)
optimizer = optim.AdamW(model.parameters(), lr=lr)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.98)
mse = nn.MSELoss()

setup_logging(run_name, current_dir)

diffusion = Diffusion(noise_steps=model.num_timesteps, batch_size=batch_size, vocab_size=model.vocab_size,
                      time_steps=model.seq_len)


train_losses = []

current_dir = os.getcwd()
# to_save_dir = "/storage/local/ssd/zigakleine-workspace"
to_save_dir = os.getcwd()

imgs_generated = []


if training_data_type == "img":
    image = Image.open('./img.jpg')
    # Convert the image to a NumPy array
    image = image.convert('L')
    imgarr = np.array(np.array(image)/255, dtype=np.float32)
    imgarr_out = np.array(imgarr * 255, dtype=np.uint8)
    im_out = Image.fromarray(imgarr_out)
    im_out.save("img_gray.jpg")
    imgarr = imgarr[None, :, :]
    imgarr_n = F.normalize(torch.tensor(imgarr), (0.5,), (0.5,), False)
elif training_data_type == "song":
    songs_path = "./mario_encoded.pkl"
    songs = pickle.load(open(songs_path, "rb"))
    song = songs[0]

    song_split = np.split(song, 4, axis=0)
    song_hstacked = np.hstack(song_split)

    song = song_hstacked[None, :, :]
    song = normalize_dataset(torch.tensor(song), dmin, dmax, None)

for epoch in range(epochs_num):

    logging.info(f"Starting epoch{epoch}:")

    train_count = 0
    train_loss_sum = 0

    emotions = None
    if training_data_type == "img":
        batch = imgarr_n.to(device)
    elif training_data_type == "song":
        batch = song.to(device)

    t = diffusion.sample_timesteps(1).to(device)

    x_t, noise = diffusion.noise_latents(batch, t)
    predicted_noise = model(x_t, t, emotions)

    loss = mse(noise, predicted_noise)
    train_loss = loss.item()

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    scheduler.step()

    train_losses.append(train_loss)
    current_lr = optimizer.param_groups[0]['lr']
    logging.info(f"Learning rate at epoch  epoch:{current_lr}")
    logging.info(f"Epoch {epoch} mean training loss: {train_loss}")

    if epoch % 1000 == 0:
        epochs = range(len(train_losses))
        plt.plot(epochs, train_losses, 'r', label='Training Loss')
        # Add labels and a legend
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Validation and Training Losses')
        plt.legend()
        loss_plot_abs_path = os.path.join(to_save_dir, "results", run_name, "graphs", f"loss_plot_{epoch}.png")
        plt.savefig(loss_plot_abs_path)
        plt.clf()

    if epoch % 1000 == 0:

        sampled_latents = diffusion.sample(model, 1, None, cfg_scale=0)

        if training_data_type == "img":
            sampled_latents = (sampled_latents.clamp(-1, 1) + 1) / 2
            sampled_latents = (sampled_latents * 255).type(torch.uint8)

            sampled_latents = torch.Tensor.cpu(sampled_latents).numpy()
            sampled_latents = sampled_latents.squeeze(0)
            imgs_generated.append(sampled_latents)
            im = Image.fromarray(sampled_latents)
            generated_abs_path = os.path.join(to_save_dir, "results", run_name, "generated", f"ep_{epoch}.jpg")
            im.save(generated_abs_path)

        elif training_data_type == "song":

            batch_transformed = inverse_data_transform(torch.Tensor.cpu(sampled_latents), dmin, dmax, None)
            generated_batch_abs_path = os.path.join(to_save_dir, "results", run_name, "generated", f"{epoch}_epoch_batch.pkl")
            file = open(generated_batch_abs_path, 'wb')
            pickle.dump(batch_transformed, file)
            file.close()


if training_data_type == "img":
    imgall = Image.fromarray(np.hstack(imgs_generated))
    generated_all_abs_path = os.path.join(to_save_dir, "results", run_name, "generated", f"all.jpg")
    imgall.save(generated_all_abs_path)
