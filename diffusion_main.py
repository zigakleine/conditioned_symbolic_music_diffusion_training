

from datetime import datetime
import pickle
import os
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np

from torch.utils.data import DataLoader

import logging
from models.transformer_film_emotion import TransformerDDPME, count_parameters
from lakh_dataset import LakhMidiDataset
from nesmdb_dataset import NesmdbMidiDataset
import json

from torch.utils.tensorboard import SummaryWriter

logging.basicConfig(format="%(asctime)s - %(levelname)s: %(message)s", level=logging.INFO, datefmt="%I:%M:%S")


class Diffusion:

    def __init__(self, noise_steps=1000, beta_start=1e-4, beta_end=0.02, batch_size=64, vocab_size=2048, time_steps=16):

        self.noise_steps = noise_steps
        self.beta_start = beta_start
        self.beta_end = beta_end

        self.batch_size = batch_size
        self.vocab_size = vocab_size
        self.time_steps = time_steps

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # linear noise schedule (u can use cosine)
        self.beta = self.prepare_noise_schedule().to(self.device)
        self.alpha = 1.0 - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)

    def prepare_noise_schedule(self):
        return torch.linspace(self.beta_start, self.beta_end, self.noise_steps)

    def noise_latents(self, x, t):

        t_squeeze = t.squeeze(-1)
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t_squeeze])[:, None, None]
        sqrt_one_minus_alpha_hat = torch.sqrt(1 - self.alpha_hat[t_squeeze])[:, None, None]
        eps = torch.randn_like(x)

        # test = sqrt_alpha_hat * x
        # test2 = sqrt_one_minus_alpha_hat * eps

        return sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * eps, eps

    def sample_timesteps(self, n):
        return torch.randint(low=1, high=self.noise_steps, size=(n, 1))

    def sample(self, model, n, emotion, cfg_scale=3):
        logging.info(f"sampling {n} new latents...")
        model.eval()
        with torch.no_grad():
            x = torch.randn((n, self.time_steps, self.vocab_size)).to(self.device)
            for i in tqdm(reversed(range(1, self.noise_steps)), position=0):
            # for i in reversed(range(1, self.noise_steps)):
                t = (torch.ones(n)*i).long().to(self.device)
                t_expand = t[:, None]
                predicted_noise = model(x, t_expand, emotion)
                if cfg_scale > 0:
                    uncond_predicted_noise = model(x, t_expand, None)
                    predicted_noise = torch.lerp(uncond_predicted_noise, predicted_noise, cfg_scale)

                alpha = self.alpha[t][:, None, None]
                alpha_hat = self.alpha_hat[t][:, None, None]
                beta = self.beta[t][:, None, None]
                if i > 1:
                    noise = torch.randn_like(x)
                else:
                    noise = torch.zeros_like(x)

                x = 1 / torch.sqrt(alpha) * (x - ((1 - alpha) / (torch.sqrt(1 - alpha_hat))) * predicted_noise) + torch.sqrt(beta) * noise

            model.train()
            return x


def setup_logging(run_name, current_dir):
    os.makedirs(os.path.join(current_dir, "checkpoints"), exist_ok=True)
    os.makedirs(os.path.join(current_dir, "results"), exist_ok=True)
    os.makedirs(os.path.join(current_dir, "checkpoints", run_name), exist_ok=True)
    os.makedirs(os.path.join(current_dir, "results", run_name), exist_ok=True)
    os.makedirs(os.path.join(current_dir, "results", run_name, "generated"), exist_ok=True)
    os.makedirs(os.path.join(current_dir, "results", run_name, "graphs"), exist_ok=True)

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

def choose_labels(l, is_lakh):

    if is_lakh:
        genres = torch.tensor([-1], dtype=torch.int64)
        composers = torch.tensor([-1], dtype=torch.int64)
    else:
        if np.random.random() < 0.1:
            genres = torch.tensor([-1], dtype=torch.int64)
            composers = torch.tensor([-1], dtype=torch.int64)
        else:
            label_choice = np.random.choice([0, 1, 2], 1)
            if label_choice[0] == 0:
                genres = l[0]
                composers = torch.tensor([-1], dtype=torch.int64)
            elif label_choice[0] == 1:
                genres = torch.tensor([-1], dtype=torch.int64)
                composers = l[1]
            elif label_choice[0] == 2:
                genres = l[0]
                composers = l[1]

    return genres, composers


def choose_labels_emotion(l, is_lakh):

    if is_lakh:
        emotions = None
    else:
        if np.random.random() < 0.1:
            emotions = None
        else:
            emotions = l

    return emotions


def train():


    now = datetime.now()
    formatted = now.strftime("%Y-%m-%d %H:%M:%S")
    print("started training at:", formatted)

    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        for i in range(num_gpus):
            gpu_name = torch.cuda.get_device_name(i)
            print(f"GPU {i}: {gpu_name}")

    lr = 1.81e-5
    batch_size = 1
    current_dir = os.getcwd()
    to_save_dir = "/storage/local/ssd/zigakleine-workspace"
    # to_save_dir = os.getcwd()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # categories_path = "./db_metadata/nesmdb/nesmdb_categories.pkl"
    # categories_indices = pickle.load(open(categories_path, "rb"))
    # emotions_num = len(categories_indices["emotions"].keys())
    categories = {"emotions": 4}

    model = TransformerDDPME(categories).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5*(100127//batch_size), gamma=0.98)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.98)

    mse = nn.MSELoss()

    is_lakh = False
    continue_training = False
    start_from_pretrained_model = False


    if is_lakh:
        run_name = "ddpm_lakh"

    else:
        run_name = "ddpm_nesmdb_1310_overfittest"

    if start_from_pretrained_model:
        existing_model_run_name = "ddpm_lakh"

        existing_model_abs_path = os.path.join(current_dir, "checkpoints", existing_model_run_name,
                                               "min_checkpoint.pth.tar")
        checkpoint = torch.load(existing_model_abs_path)
        model.load_state_dict(checkpoint["state_dict"])

        print(f"starting from pretrained lakh model {existing_model_abs_path}")
    else:
        if continue_training:
            existing_model_run_name = run_name
            existing_model_abs_path = os.path.join(current_dir, "checkpoints", existing_model_run_name,
                                                   "last_checkpoint.pth.tar")
            checkpoint = torch.load(existing_model_abs_path)

            model.load_state_dict(checkpoint["state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer"])
            epoch_num_start = checkpoint["epoch"]
            min_val_loss_start = checkpoint["min_val_loss"]
            print(f"loaded existing model {existing_model_abs_path}, at epoch {epoch_num_start}, atl val loss {min_val_loss_start}")

        else:
            min_val_loss_start = float("inf")
            print(f"starting from zero")


    setup_logging(run_name, to_save_dir)
    diffusion = Diffusion(noise_steps=model.num_timesteps, batch_size=batch_size, vocab_size=model.vocab_size, time_steps=model.seq_len)

    print("device is", diffusion.device)
    print("dataset is", run_name)
    print("continue_training", continue_training)
    print("starting from lakh", start_from_pretrained_model)


    epochs_num = 20000

    run_info_params = {
        "run_name": run_name,
        "continue_training": continue_training,
        "start_from_lakh": start_from_pretrained_model,
        "seq_len": model.seq_len,
        "vocab_size": model.vocab_size,
        "num_timesteps": model.num_timesteps,
        "embed_size": model.embed_size,
        "num_heads": model.num_heads,
        "num_layers": model.num_layers,
        "num_mlp_layers": model.num_mlp_layers,
        "mlp_dims": model.mlp_dims,
        "beta_start": diffusion.beta_start,
        "beta_end": diffusion.beta_end,
        "lr": lr,
        "batch_size": batch_size,
        "trainable_params": count_parameters(model),
    }
    rip = json.dumps(run_info_params, indent=4)
    params_abs_path = os.path.join(to_save_dir, "results", run_name, "run_info_params.json")
    file_json = open(params_abs_path, 'w')
    file_json.write(rip)
    file_json.close()

    std_devs_tracks = pickle.load(open("./std_devs_singletrack_2.pkl", "rb"))
    std_devs_masks = []
    num_latents = 42

    for std_dev_track in std_devs_tracks:
        std_dev_track = std_dev_track[:num_latents]
        std_dev_idx_track = [i for i, dev in std_dev_track]
        std_dev_idx_track = np.array(std_dev_idx_track)

        std_dev_mask = np.zeros((512,), dtype=bool)
        std_dev_mask[std_dev_idx_track] = True

        std_devs_masks.append(std_dev_mask)

    #load data

    if is_lakh:
        dataset = LakhMidiDataset(transform=normalize_dataset, std_dev_masks=std_devs_masks)
        train_ds, test_ds = torch.utils.data.random_split(dataset, [272702, 8434])

    else:
        dataset = NesmdbMidiDataset(transform=normalize_dataset, std_dev_masks=std_devs_masks)
        # train_ds, test_ds = torch.utils.data.random_split(dataset, [100127, 3097])
        # train_ds, test_ds = torch.utils.data.random_split(dataset, [1, 4])

    train_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)
    # test_loader = DataLoader(dataset=test_ds, batch_size=batch_size, shuffle=True)

    if continue_training:
        train_losses_abs_path = os.path.join(to_save_dir, "results", existing_model_run_name, "train_losses.pkl")
        train_losses = pickle.load(open(train_losses_abs_path, "rb"))

        val_losses_abs_path = os.path.join(to_save_dir, "results", existing_model_run_name, "val_losses.pkl")
        val_losses = pickle.load(open(val_losses_abs_path, "rb"))

        starting_epoch = len(train_losses)
        min_val_loss = min_val_loss_start
    else:
        val_losses = []
        train_losses = []
        starting_epoch = 0
        min_val_loss = min_val_loss_start

    for epoch in range(epochs_num):

        logging.info(f"Starting epoch {starting_epoch + epoch}:")
        # pbar = tqdm(train_loader)
        pbar = train_loader

        train_count = 0
        train_loss_sum = 0

        for step, (batch, l) in enumerate(pbar):

            # emotions = choose_labels_emotion(l, is_lakh)
            emotions = None
            if emotions is not None:
                emotions = emotions.to(device)
            batch = batch.to(device)
            t = diffusion.sample_timesteps(batch.shape[0]).to(device)

            x_t, noise = diffusion.noise_latents(batch, t)
            predicted_noise = model(x_t, t, emotions)

            loss = mse(noise, predicted_noise)
            train_loss_sum += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            train_count += 1

        mean_train_loss = train_loss_sum / train_count
        train_losses.append(mean_train_loss)
        current_lr = optimizer.param_groups[0]['lr']
        logging.info(f"Learning rate at epoch  {starting_epoch + epoch}:{current_lr}")
        logging.info(f"Epoch {starting_epoch + epoch} mean training loss: {mean_train_loss}")
        val_count = 0
        val_loss_sum = 0
        logging.info(f"Validation for epoch {starting_epoch + epoch}:")

        # pbar_test = test_loader
        # pbar_test = tqdm(test_loader)
        # model.eval()
        # with torch.no_grad():
        #
        #     for step, (batch, l) in enumerate(pbar_test):
        #
        #         emotions = choose_labels_emotion(l, is_lakh)
        #
        #         if emotions is not None:
        #             emotions = emotions.to(device)
        #         batch = batch.to(device)
        #
        #         t = diffusion.sample_timesteps(batch.shape[0]).to(device)
        #
        #         x_t, noise = diffusion.noise_latents(batch, t)
        #
        #         predicted_noise = model(x_t, t, emotions)
        #         val_loss = mse(noise, predicted_noise)
        #         val_loss_sum += val_loss.item()
        #
        #         val_count += 1
        #
        #     mean_val_loss = val_loss_sum / val_count
        #     val_losses.append(mean_val_loss)
        #     logging.info(f"Epoch {starting_epoch + epoch} mean validation loss: {mean_val_loss}")
        # model.train()

        # if mean_val_loss < min_val_loss:
        #     min_val_loss = mean_val_loss
        #     logging.info(f"!!! New min validation loss at epoch {starting_epoch + epoch}, mean validation loss: {mean_val_loss}")
        #     min_model_abs_path = os.path.join(to_save_dir, "checkpoints", run_name, "min_checkpoint.pth.tar")
        #     checkpoint = {"state_dict": model.state_dict(), "optimizer": optimizer.state_dict(),
        #                   "epoch": (starting_epoch + epoch), "min_val_loss": min_val_loss}
        #     torch.save(checkpoint, min_model_abs_path)
        #
        if epoch % 1000 == 0:
            sampled_latents = diffusion.sample(model, 1, None, cfg_scale=0)
            batch_transformed = inverse_data_transform(torch.Tensor.cpu(sampled_latents), -14., 14., std_devs_masks)

            generated_batch_abs_path = os.path.join(to_save_dir, "results", run_name, "generated", f"{starting_epoch + epoch}_epoch_batch.pkl")
            file = open(generated_batch_abs_path, 'wb')
            pickle.dump(batch_transformed, file)
            file.close()

        # checkpoint = {"state_dict": model.state_dict(), "optimizer": optimizer.state_dict(),
        #               "epoch": (starting_epoch + epoch), "min_val_loss": min_val_loss}
        # min_model_abs_path = os.path.join(to_save_dir, "checkpoints", run_name, "last_checkpoint.pth.tar")
        # torch.save(checkpoint, min_model_abs_path)

        # if epoch == 99:
        #     checkpoint = {"state_dict": model.state_dict(), "optimizer": optimizer.state_dict(),
        #                   "epoch": (starting_epoch + epoch), "min_val_loss": min_val_loss}
        #     hund_model_abs_path = os.path.join(to_save_dir, "checkpoints", run_name, "100_checkpoint.pth.tar")
        #     torch.save(checkpoint, hund_model_abs_path)
        #
        # if epoch == 149:
        #     checkpoint = {"state_dict": model.state_dict(), "optimizer": optimizer.state_dict(),
        #                   "epoch": (starting_epoch + epoch), "min_val_loss": min_val_loss}
        #     hundten_model_abs_path = os.path.join(to_save_dir, "checkpoints", run_name, "150_checkpoint.pth.tar")
        #     torch.save(checkpoint, hundten_model_abs_path)

        #  picklaj losse

        # train_losses_abs_path = os.path.join(to_save_dir, "results", run_name, "train_losses.pkl")
        # file = open(train_losses_abs_path, 'wb')
        # pickle.dump(train_losses, file)
        # file.close()
        #
        # val_losses_abs_path = os.path.join(to_save_dir, "results", run_name, "val_losses.pkl")
        # file = open(val_losses_abs_path, 'wb')
        # pickle.dump(val_losses, file)
        # file.close()
        #

        if epoch % 1000 == 0:
        # Plot validation losses in blue and training losses in red
            epochs = range(len(train_losses))
            # plt.plot(epochs, val_losses, 'b', label='Validation Loss')
            plt.plot(epochs, train_losses, 'r', label='Training Loss')

            # Add labels and a legend
            plt.xlabel('Epochs')
            plt.ylabel('Loss')
            plt.title('Validation and Training Losses')
            plt.legend()
            loss_plot_abs_path = os.path.join(to_save_dir, "results", run_name, "graphs", f"loss_plot_{starting_epoch+epoch}.png")
            plt.savefig(loss_plot_abs_path)
            plt.clf()

    now = datetime.now()
    formatted = now.strftime("%Y-%m-%d %H:%M:%S")
    print("ended training at:", formatted)



if __name__ == "__main__":
    train()
