""" Training Implementation for RSA
"""

import torch
import torch.nn as nn
import yaml

from data.utils import *
from data.dataset import SAdataset, get_data_loader, infinite_iter

from models.RSA import RSA
from models.ACG import ACG
from models.utils import *


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
key_len = config["RSA"]["struct"]["basic"]["key_len"]
batch_size = config["RSA"]["training"]["batch_size"]
num_workers = config["RSA"]["training"]["num_workers"]
acg_cond = cc(
    torch.tensor(
        [
            [
                0,
            ]
        ]
    ).to(torch.float32)
)
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
acg = cc(ACG(config))
rsa = cc(RSA(config))
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
criterion_consistance = nn.MSELoss()
# triplet loss


"""##########################################Training
"""
print("[RSA]Starting training... ")
n_iterations = config["RSA"]["training"]["n_iterations"]
summary_steps = config["RSA"]["training"]["summary_steps"]
autosave_steps = config["RSA"]["training"]["autosave_steps"]

consistance_loss_history = []
triplet_loss_history = []
restore_loss_history = []
lambda_1 = config["RSA"]["training"]["lambda_1"]
lambda_2 = config["RSA"]["training"]["lambda_2"]
lambda_3 = config["RSA"]["training"]["lambda_3"]

for iter in range(n_iterations):
    # get data
    orig_spk_emb, mel = next(train_iter)
    orig_spk_emb = orig_spk_emb.to(torch.float32)
    mel = mel.to(torch.float32)
    key = torch.randn(batch_size, 1, key_len).to(torch.float32)
    # load to device
    orig_spk_emb = cc(orig_spk_emb)
    mel = cc(mel)
    key = cc(key)
    # get condition
    cond, _ = acg(key, acg_cond)
    # forward & backward processes of cINN
    mel_ano = rsa(mel, cond)
    mel_res = rsa(mel_ano, cond, True)
    mel_recon = rsa(mel, orig_spk_emb)
    # test
    # if iter == 0:
    #     print("key.shape: ", key.shape)
    #     print("acg_cond.shape: ", acg_cond.shape)
    #     print("cond.shape: ", cond.shape)
    #     print("orig_spk_emb.shape: ", orig_spk_emb.shape)
    #     print("mel.shape: ", mel.shape)
    #     print("mel_ano.shape: ", mel_ano.shape)
    #     print("mel_res.shape: ", mel_res.shape)
    #     print("mel_recon.shape: ", mel_recon.shape)
    #     print("mel", mel[0][0][:10])
    #     print("mel_res", mel_res[0][0][:10])
    #     print("mel_ano", mel_ano[0][0][:10])
    #     print("mel_recon", mel_recon[0][0][:10])
    # calculate losses
    consistance_loss = criterion_consistance(mel, mel_recon)
    # restore_loss = criterion_restore(mel, mel_res)
    triplet_loss = 0
    # save losses
    consistance_loss_history.append(consistance_loss)
    # restore_loss_history.append(restore_loss)
    triplet_loss_history.append(triplet_loss)
    # total loss
    total_loss = lambda_1 * consistance_loss  # + lambda_2 * triplet_loss
    # nn backward
    total_loss.backward()
    optim.step()
    optim.zero_grad()
    # logging
    print(
        f"[RSA]:[{iter+1}/{n_iterations}]",
        f"loss_consistance={consistance_loss.item():6f}",
        # f"loss_ident={triplet_loss_loss.item():6f}",
        end="\r",
    )
    # summary
    if (iter + 1) % summary_steps == 0 or iter + 1 == n_iterations:
        print()
    # autosave
    if (iter + 1) % autosave_steps == 0 or iter + 1 == n_iterations:
        torch.save(rsa.state_dict(), f"{save_path}/rsa.ckpt")
        torch.save(optim.state_dict(), f"{save_path}/optim.opt")
