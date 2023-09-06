

from datetime import datetime
import pickle
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np

from torch.utils.data import DataLoader

import logging
from models.transformer_film import TransformerDDPM
from lakh_dataset import LakhMidiDataset
from nesmdb_dataset import NesmdbMidiDataset

from torch.utils.tensorboard import SummaryWriter

logging.basicConfig(format="%(asctime)s - %(levelname)s: %(message)s", level=logging.INFO, datefmt="%I:%M:%S")


class Diffusion:

    def __init__(self, noise_steps=1000, beta_start=1e-4, beta_end=0.02, batch_size=64, vocab_size=76, time_steps=32):



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

    def sample(self, model, n, genres, composers, cfg_scale=3):
        logging.info(f"sampling {n} new latents...")
        model.eval()
        with torch.no_grad():
            x = torch.randn((n, self.time_steps, self.vocab_size)).to(self.device)
            for i in tqdm(reversed(range(1, self.noise_steps)), position=0):
            # for i in reversed(range(1, self.noise_steps)):
                t = (torch.ones(n)*i).long().to(self.device)
                t_expand = t[:, None]
                predicted_noise = model(x, t_expand, genres, composers)
                if cfg_scale > 0:
                    uncond_predicted_noise = model(x, t_expand, torch.tensor([-1], dtype=torch.int64), torch.tensor([-1], dtype=torch.int64))
                    predicted_noise = torch.lerp(uncond_predicted_noise, predicted_noise)

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



def setup_logging(run_name):
    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    os.makedirs(os.path.join("checkpoints", run_name), exist_ok=True)
    os.makedirs(os.path.join("results", run_name), exist_ok=True)
    os.makedirs(os.path.join("results", run_name, "generated"), exist_ok=True)
    os.makedirs(os.path.join("results", run_name, "graphs"), exist_ok=True)

def normalize_dataset(batch, data_min, data_max):
    """Normalize dataset to range [-1, 1]."""
    batch = (batch - data_min) / (data_max - data_min)
    batch = 2. * batch - 1.
    return batch

def inverse_data_transform(batch, slices, data_min, data_max):
    out_channels = 512
    batch = batch.numpy()
    batch = (batch + 1.) / 2.
    batch = (data_max - data_min) * batch + data_min

    transformed = np.random.randn(*batch.shape[:-1], out_channels)
    transformed[..., slices] = batch
    batch = transformed
    return batch

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



def train():


    now = datetime.now()
    formatted = now.strftime("%Y-%m-%d %H:%M:%S")
    print("started training at:", formatted)

    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        for i in range(num_gpus):
            gpu_name = torch.cuda.get_device_name(i)
            print(f"GPU {i}: {gpu_name}")

    lr = 1e-3
    batch_size = 64
    current_dir = os.getcwd()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    categories_path = "./db_metadata/nesmdb/nesmdb_categories.pkl"
    categories_indices = pickle.load(open(categories_path, "rb"))
    genres_num = len(categories_indices["genres"].keys())
    composers_num = len(categories_indices["composers"].keys())
    categories = {"genres": genres_num, "composers": composers_num}

    model = TransformerDDPM(categories).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.98)
    mse = nn.MSELoss()

    is_lakh = True
    load_existing = False
    start_at_epoch_zero = True

    if is_lakh:
        run_name = "ddpm_lakh"
        min_max_ckpt_path = "./pkl_info/lakh_min_max.pkl"
    else:
        run_name = "ddpm_nesmdb"
        min_max_ckpt_path = "./pkl_info/nesmdb_min_max.pkl"

    existing_model_run_name = "ddpm_lakh"
    existing_model_abs_path = os.path.join(current_dir, "checkpoints", existing_model_run_name, "last_checkpoint.pth.tar")

    if load_existing:
        checkpoint = torch.load(existing_model_abs_path)
        model.load_state_dict(checkpoint["state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        epoch_num = checkpoint["epoch"]

    setup_logging(run_name)
    diffusion = Diffusion()

    print("device is", diffusion.device)
    print("dataset is", run_name)
    print("load_existing", load_existing)
    print("start_at_epoch_zero", start_at_epoch_zero)

    slice_ckpt = "./pkl_info/fb256_slices_76.pkl"
    fb256_slices = pickle.load(open(slice_ckpt, "rb"))
    min_max = pickle.load(open(min_max_ckpt_path, "rb"))

    epochs = 120

    #load data

    if is_lakh:
        dataset = LakhMidiDataset(min_max=min_max, transform=normalize_dataset)
        train_ds, test_ds = torch.utils.data.random_split(dataset, [272534, 8428])
    else:
        dataset = NesmdbMidiDataset(min_max=min_max, transform=normalize_dataset)
        train_ds, test_ds = torch.utils.data.random_split(dataset, [100127, 3097])

    train_loader = DataLoader(dataset=train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_ds, batch_size=batch_size, shuffle=True)

    if load_existing and not start_at_epoch_zero:
        train_losses_abs_path = os.path.join(current_dir, "results", existing_model_run_name, "train_losses.pkl")
        train_losses = pickle.load(open(train_losses_abs_path, "rb"))

        val_losses_abs_path = os.path.join(current_dir, "results", existing_model_run_name, "val_losses.pkl")
        val_losses = pickle.load(open(val_losses_abs_path, "rb"))

        starting_epoch = len(train_losses)
    else:
        val_losses = []
        train_losses = []

        starting_epoch = 0

    min_val_loss = float("inf")

    for epoch in range(epochs):

        logging.info(f"Starting epoch {starting_epoch + epoch}:")
        pbar = tqdm(train_loader)
        # pbar = train_loader

        train_count = 0
        train_loss_sum = 0

        for step, (batch, l) in enumerate(pbar):
            # print(step, batch[0])

            genre, composer = choose_labels(l, is_lakh)
            genre = genre.to(device)
            composer = composer.to(device)
            batch = batch.to(device)

            t = diffusion.sample_timesteps(batch.shape[0]).to(device)

            x_t, noise = diffusion.noise_latents(batch, t)
            predicted_noise = model(x_t, t, genre, composer)

            loss = mse(noise, predicted_noise)
            train_loss_sum += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            train_count +=1

        mean_train_loss = train_loss_sum / train_count
        train_losses.append(mean_train_loss)
        current_lr = optimizer.param_groups[0]['lr']
        logging.info(f"Learning rate at epoch  {starting_epoch + epoch}:{current_lr}")
        logging.info(f"Epoch {starting_epoch + epoch} mean training loss: {mean_train_loss}")
        val_count = 0
        val_loss_sum = 0
        logging.info(f"Validation for epoch {starting_epoch + epoch}:")

        # pbar_test = test_loader
        pbar_test = tqdm(test_loader)

        with torch.no_grad():

            for step, (batch, l) in enumerate(pbar_test):

                genre, composer = choose_labels(l, is_lakh)
                genre = genre.to(device)
                composer = composer.to(device)
                batch = batch.to(device)

                t = diffusion.sample_timesteps(batch.shape[0]).to(device)

                x_t, noise = diffusion.noise_latents(batch, t)
                predicted_noise = model(x_t, t, genre, composer)
                val_loss = mse(noise, predicted_noise)
                val_loss_sum += val_loss.item()

                val_count += 1

            mean_val_loss = val_loss_sum / val_count
            val_losses.append(mean_val_loss)
            logging.info(f"Epoch {starting_epoch + epoch} mean validation loss: {mean_val_loss}")

        if mean_val_loss < min_val_loss:
            min_val_loss = mean_val_loss
            logging.info(f"!!! New min validation loss at epoch {starting_epoch + epoch}, mean validation loss: {mean_val_loss}")
            min_model_abs_path = os.path.join(current_dir, "checkpoints", run_name, "min_checkpoint.pth.tar")
            checkpoint = {"state_dict": model.state_dict(), "optimizer": optimizer.state_dict(), "epoch": (starting_epoch + epoch)}
            torch.save(checkpoint, min_model_abs_path)

        sampled_latents = diffusion.sample(model, 1, torch.tensor([-1], dtype=torch.int64), torch.tensor([-1], dtype=torch.int64), cfg_scale=0)
        batch_transformed = inverse_data_transform(torch.Tensor.cpu(sampled_latents), fb256_slices, min_max["min"],
                                                   min_max["max"])
        generated_batch_abs_path = os.path.join(current_dir, "results", run_name, "generated", f"{starting_epoch + epoch}_epoch_batch.pkl")
        file = open(generated_batch_abs_path, 'wb')
        pickle.dump(batch_transformed, file)
        file.close()

        checkpoint = {"state_dict": model.state_dict(), "optimizer": optimizer.state_dict(), "epoch": (starting_epoch + epoch)}
        min_model_abs_path = os.path.join(current_dir, "checkpoints", run_name, "last_checkpoint.pth.tar")
        torch.save(checkpoint, min_model_abs_path)

        if epoch == 99:
            checkpoint = {"state_dict": model.state_dict(), "optimizer": optimizer.state_dict(),
                          "epoch": (starting_epoch + epoch)}
            hund_model_abs_path = os.path.join(current_dir, "checkpoints", run_name, "100_checkpoint.pth.tar")
            torch.save(checkpoint, hund_model_abs_path)

        if epoch == 109:
            checkpoint = {"state_dict": model.state_dict(), "optimizer": optimizer.state_dict(),
                          "epoch": (starting_epoch + epoch)}
            hundten_model_abs_path = os.path.join(current_dir, "checkpoints", run_name, "110_checkpoint.pth.tar")
            torch.save(checkpoint, hundten_model_abs_path)

        #  picklaj losse

        train_losses_abs_path = os.path.join(current_dir, "results", run_name, "train_losses.pkl")
        file = open(train_losses_abs_path, 'wb')
        pickle.dump(train_losses, file)
        file.close()

        val_losses_abs_path = os.path.join(current_dir, "results", run_name, "val_losses.pkl")
        file = open(val_losses_abs_path, 'wb')
        pickle.dump(val_losses, file)
        file.close()

        # Plot validation losses in blue and training losses in red
        epochs = range(len(val_losses))
        plt.plot(epochs, val_losses, 'b', label='Validation Loss')
        plt.plot(epochs, train_losses, 'r', label='Training Loss')

        # Add labels and a legend
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Validation and Training Losses')
        plt.legend()
        loss_plot_abs_path = os.path.join(current_dir, "results", run_name, "graphs", f"loss_plot_{starting_epoch+epoch}.png")
        plt.savefig(loss_plot_abs_path)

    now = datetime.now()
    formatted = now.strftime("%Y-%m-%d %H:%M:%S")
    print("ended training at:", formatted)

if __name__ == "__main__":
    train()
