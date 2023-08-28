
#import tensorflow as tf
import tensorflow_datasets as tfds
import input_pipeline

import os
import torch
import torch.nn as nn
from torch import optim
from tqdm import tqdm

import logging
from models.transformer_film import TransformerDDPM

from torch.utils.tensorboard import SummaryWriter

logging.basicConfig(format="%(asctime)s - %(levelname)s: %(message)s", level=logging.INFO, datefmt="%I:%M:%S")


class Diffusion:

    def __init__(self, noise_steps=1000, beta_start=1e-4, beta_end=0.02, batch_size=64, vocab_size=42, time_steps=32,
                 device='cpu'):

        self.device = device

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

    def sample(self, model, n):
        logging.info(f"sampling {n} new latents...")
        model.eval()
        with torch.no_grad():
            x = torch.randn((n, self.time_steps, self.vocab_size)).to(self.device)
            for i in tqdm(reversed(range(1, self.noise_steps)), position=0):
                t = (torch.ones(n)*i).long().to(self.device)
                t_expand = t[:, None]
                predicted_noise = model(x, t_expand)

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
    os.makedirs("models", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    os.makedirs(os.path.join("models", run_name), exist_ok=True)
    os.makedirs(os.path.join("results", run_name), exist_ok=True)


def train():

    print("starting")

    #tf.config.experimental.set_visible_devices([], 'GPU')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_name = "DDPM_unconditional"
    setup_logging(run_name)
    lr = 1e-3
    model = TransformerDDPM().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    mse = nn.MSELoss()
    diffusion = Diffusion(device=device)
    # logger = SummaryWriter(os.path.join("runs",  run_name))
    print("device is" , diffusion.device)

    dataset = "/content/drive/MyDrive/notesequences"
    # dataset = "./training_data"
    data_shape = (32, 512)
    problem = 'vae'
    batch_size = 64
    normalize = True
    slice_ckpt='./checkpoints/slice-mel-512.pkl'
    include_cardinality = False

    epochs = 150

    train_ds, test_ds = input_pipeline.get_dataset(dataset, data_shape, batch_size, normalize, slice_ckpt, include_cardinality)

    val_losses = []
    train_losses = []
    train_numpy = tfds.as_numpy(train_ds)
    test_numpy = tfds.as_numpy(test_ds)


    for epoch in range(epochs):

        logging.info(f"Starting epoch {epoch}:")
        pbar = tqdm(train_numpy)
        train_count = 0
        train_loss_sum = 0

        for step, batch in enumerate(pbar):
            # print(step, batch[0])

            batch = torch.from_numpy(batch).to(device)
            t = diffusion.sample_timesteps(batch.shape[0]).to(device)

            x_t, noise = diffusion.noise_latents(batch, t)
            predicted_noise = model(x_t, t)

            loss = mse(noise, predicted_noise)
            train_loss_sum += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_count +=1

        mean_train_loss = train_loss_sum / train_count
        train_losses.append(mean_train_loss)
        logging.info(f"Epoch {epoch} mean training loss: {mean_train_loss}")
        val_count = 0
        val_loss_sum = 0

        with torch.no_grad():
            for step, batch in enumerate(test_numpy):
                batch = torch.from_numpy(batch).to(device)
                t = diffusion.sample_timesteps(batch.shape[0]).to(device)

                x_t, noise = diffusion.noise_latents(batch, t)
                predicted_noise = model(x_t, t)
                val_loss = mse(noise, predicted_noise)
                val_loss_sum += val_loss.item()

                val_count += 1

            mean_val_loss = val_loss_sum / val_count
            val_losses.append(mean_val_loss)
            logging.info(f"Epoch {epoch} mean validation loss: {mean_val_loss}")

        sampled_latents = diffusion.sample(model, 1)
        batch_transformed = input_pipeline.inverse_data_transform(torch.Tensor.cpu(sampled_latents), slice_ckpt=slice_ckpt,
                                                                  data_min=train_ds.min, data_max=train_ds.max)

        torch.save(batch_transformed, os.path.join(dataset, "results", run_name, f"{epoch}_epoch_batch.pt"))
        if epoch % 10 == 0:
            torch.save(model.state_dict(), os.path.join(dataset, "models", run_name, f"{epoch}_model_ckpt.pt"))

    torch.save(train_losses, os.path.join(dataset, "results", run_name, f"train_losses.pt"))
    torch.save(val_losses, os.path.join(dataset,  "results", run_name, f"val_losses.pt"))

if __name__ == "__main__":
    train()
