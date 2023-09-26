
from torch.utils.data import DataLoader
from nesmdb_dataset import NesmdbMidiDataset
from lakh_dataset import LakhMidiDataset
import torch
import pickle

def normalize_dataset(batch, data_min, data_max):
    """Normalize dataset to range [-1, 1]."""
    batch = (batch - data_min) / (data_max - data_min)
    # batch = 2. * batch - 1.
    return batch


dataset = "nesmdb"
# dataset = "lakh"


if dataset == "nesmdb":
    min_max_ckpt_path = "./pkl_info/nesmdb_min_max.pkl"
elif dataset == "lakh":
    min_max_ckpt_path = "./pkl_info/lakh_min_max.pkl"

batch_size = 64
min_max = pickle.load(open(min_max_ckpt_path, "rb"))


if dataset == "nesmdb":
    dataset = NesmdbMidiDataset(min_max=min_max, transform=normalize_dataset)
    train_ds, test_ds = torch.utils.data.random_split(dataset, [100127, 3097])
elif dataset == "lakh":
    dataset = LakhMidiDataset(min_max=min_max, transform=normalize_dataset)
    train_ds, test_ds = torch.utils.data.random_split(dataset, [272702, 8434])


train_loader = DataLoader(dataset=train_ds, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_ds, batch_size=batch_size, shuffle=True)

for step, (batch, l) in enumerate(train_loader):
    print("shape", batch.shape)