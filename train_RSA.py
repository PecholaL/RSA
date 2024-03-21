""" Training Implementation for RSA
"""

import torch
import torch.nn as nn
import yaml

from data.utils import *
from data.dataset import SAdataset, get_data_loader, infinite_iter

from models.RSA import RSA
from models.ACG import ACG


""" Device
"""
device = "cpu"


""" Paths to Ckpt/Pkl/Json/Yaml
"""
dataset_path = "/Users/pecholalee/Coding/SpkAno/miniSAdata_pickle/audio_mel.pkl"
model_config_path = "/Users/pecholalee/Coding/RSA/models/config.yaml"
acg_ckpt_path = "/Users/pecholalee/Coding/SpkAno/RSA_data/save/acg.ckpt"
save_path = "/Users/pecholalee/Coding/SpkAno/RSA_data/save"


""" Load Config
"""
with open(model_config_path) as f:
    config = yaml.load(f, Loader=yaml.FullLoader)


""" Prepare Data
"""
batch_size = config["RSA"]["training"]["batch_size"]
num_workers = config["RSA"]["training"]["num_workers"]
dataset = SAdataset(pickle_path=dataset_path)
dataLoader = get_data_loader(
    dataset=dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=num_workers,
    drop_last=False,
)
train_iter = infinite_iter(dataLoader)
print("[RSA]Infinite dataloader is built. ")


""" Build Model
"""
acg = ACG(config)
rsa = RSA(config)
print("[RSA]ACG and RSA is built. ")
print("[RSA]Total para count: {}. ".format(sum(x.numel() for x in rsa.parameters())))


""" Load Pretrained ACG
"""
acg.load_state_dict(torch.load(acg_ckpt_path))
print("[RSA]Pretrained ACG is loaded. ")


""" Build Optimizer
"""
beta1 = config["RSA"]["training"]["beta1"]
beta2 = config["RSA"]["training"]["beta2"]
eps = float(config["RSA"]["training"]["eps"])
lr = float(config["RSA"]["training"]["lr"])
weight_decay = float(config["RSA"]["training"]["weight_decay"])
decay_by = float(config["RSA"]["training"]["decay_by"])

param_rsa = filter(lambda p: p.requires_grad, rsa.parameters())
optim = torch.optim.Adam(
    param_rsa, lr=lr, betas=(beta1, beta2), eps=eps, weight_decay=weight_decay
)
print("[RSA]Optimizer is built. ")


""" Loss
"""
# consistant loss
criterion_consistant = nn.MSELoss()
# triplet loss
# restore loss


"""##########################################Training
"""
print("[RSA]Starting training... ")
n_iterations = config["RSA"]["training"]["n_iterations"]
summary_steps = config["RSA"]["training"]["summary_steps"]
autosave_steps = config["RSA"]["training"]["autosave_steps"]

consistant_loss_history = []
triplet_loss_history = []
restore_loss_history = []
lambda_1 = config["RSA"]["training"]["lambda_1"]
lambda_2 = config["RSA"]["training"]["lambda_2"]
lambda_3 = config["RSA"]["training"]["lambda_3"]

for iter in range(n_iterations):
    # get data
    orig_spk_emb, mel = next(train_iter)


torch.save(rsa.state_dict(), f"{save_path}/rsa.ckpt")
torch.save(optim.state_dict(), f"{save_path}/optim.opt")
