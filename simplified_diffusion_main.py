from diffusion_main import Diffusion
import tqdm
import logging
import os
import torch
from models.transformer_film_img import TransformerDDPME
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
from torchvision.transforms import functional as F
from PIL import Image


def setup_logging(run_name, current_dir):

    os.makedirs(os.path.join(current_dir, "results"), exist_ok=True)
    os.makedirs(os.path.join(current_dir, "results", run_name), exist_ok=True)
    os.makedirs(os.path.join(current_dir, "results", run_name, "generated"), exist_ok=True)
    os.makedirs(os.path.join(current_dir, "results", run_name, "graphs"), exist_ok=True)

lr = 2e-4
batch_size = 1
current_dir = os.getcwd()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
run_name = "img_overfit_test"

categories = {"emotions": 4}
model = TransformerDDPME(categories).to(device)
optimizer = optim.AdamW(model.parameters(), lr=lr)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.98)
mse = nn.MSELoss()

setup_logging(run_name, current_dir)

diffusion = Diffusion(noise_steps=model.num_timesteps, batch_size=batch_size, vocab_size=model.vocab_size,
                      time_steps=model.seq_len)

epochs_num = 10000
train_losses = []

current_dir = os.getcwd()
# to_save_dir = "/storage/local/ssd/zigakleine-workspace"
to_save_dir = os.getcwd()


image = Image.open('./img.jpg')

# Convert the image to a NumPy array
image = image.convert('L')
imgarr = np.array(np.array(image)/255, dtype=np.float32)

imgarr_out = np.array(imgarr * 255, dtype=np.uint8)
im_out = Image.fromarray(imgarr_out)
im_out.save("img_gray.jpg")

imgarr = imgarr[None, :, :]
imgarr_n = F.normalize(torch.tensor(imgarr), (0.5,), (0.5,), False)

for epoch in range(epochs_num):

    logging.info(f"Starting epoch{epoch}:")

    train_count = 0
    train_loss_sum = 0

    emotions = None
    batch = imgarr_n.to(device)
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

    if epoch % 100 == 0:
        sampled_latents = diffusion.sample(model, 1, None, cfg_scale=0)
        sampled_latents = (sampled_latents.clamp(-1, 1) + 1) / 2
        sampled_latents = (sampled_latents * 255).type(torch.uint8)

        sampled_latents = torch.Tensor.cpu(sampled_latents).numpy()
        sampled_latents = sampled_latents.squeeze(0)
        im = Image.fromarray(sampled_latents)
        generated_abs_path = os.path.join(to_save_dir, "results", run_name, "generated", f"ep_{epoch}.jpg")
        im.save(generated_abs_path)