""" Test the DataLoader, check the shapes
"""

import pickle
import yaml
from random import randint
from scipy.io.wavfile import write

from utils import *
from dataset import SAdataset, get_data_loader, infinite_iter

pickle_path = "../SpkAno/miniSAdata_pickle/audio_mel.pkl"

# Load pickle dataset
with open(pickle_path, "rb") as f:
    data = pickle.load(f)
    print(f"[test]load {len(data)} data items")

# Load data processing configure
with open("./data/config.yaml") as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

sample_rate = config["sample_rate"]

# Check DataLoader
dataset = SAdataset(pickle_path=pickle_path)
dataLoader = get_data_loader(
    dataset=dataset, batch_size=4, shuffle=False, num_workers=0, drop_last=False
)
train_iter = infinite_iter(dataLoader)
print(f"[test]built infinite dataloader")
e, m = next(train_iter)
print(f"[test]mel shape: {m.shape}")
print(f"[test]emb shape: {e.shape}")
