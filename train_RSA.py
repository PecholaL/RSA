""" Training Implementation for RSA
"""

import torch
import torch.nn as nn
import yaml
from collections import OrderedDict

from data.utils import *
from data.dataset import SAdataset, get_data_loader, infinite_iter
from spk_enc.model_bl import D_VECTOR

from models.RSA import RSA
from models.ACG import ACG
from models.utils import *

# =================================================================================== #
#                             1. Paths & Configuration                                #
# =================================================================================== #
""" Device
"""
device = "cpu"


""" Paths to Ckpt/Pkl/Json/Yaml
"""
dataset_path = "/Users/pecholalee/Coding/SpkAno/miniSAdata_pickle/audio_mel.pkl"
model_config_path = "/Users/pecholalee/Coding/RSA/models/config.yaml"
acg_ckpt_path = "/Users/pecholalee/Coding/SpkAno/RSA_data/save/acg.ckpt"
spk_enc_path = "/Users/pecholalee/Coding/RSA/pretrained_models/3000000-BL.ckpt"
save_path = "/Users/pecholalee/Coding/SpkAno/RSA_data/save"


""" Load Config
"""
with open(model_config_path) as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

# =================================================================================== #
#              2. Prepare Data & Load Pretrained Model & Build Model                  #
# =================================================================================== #
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
acg_cond = acg_cond.repeat(batch_size, 1)
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
d_vec_enc = D_VECTOR(dim_input=80, dim_cell=768, dim_emb=256)
c_checkpoint = torch.load(spk_enc_path, map_location="cpu")
new_state_dict = OrderedDict()
for key, val in c_checkpoint["model_b"].items():
    new_key = key[7:]
    new_state_dict[new_key] = val
d_vec_enc.load_state_dict(new_state_dict)

# =================================================================================== #
#                          3. Optimizer & Loss Functions                              #
# =================================================================================== #
""" Build Optimizer
"""
beta1 = config["RSA"]["training"]["beta1"]
beta2 = config["RSA"]["training"]["beta2"]
eps = float(config["RSA"]["training"]["eps"])
lr = float(config["RSA"]["training"]["lr"])
weight_decay = float(config["RSA"]["training"]["weight_decay"])
weight_step = config["RSA"]["training"]["weight_step"]
gamma = config["RSA"]["training"]["gamma"]

param_rsa = filter(lambda p: p.requires_grad, rsa.parameters())
optim = torch.optim.Adam(
    param_rsa, lr=lr, betas=(beta1, beta2), eps=eps, weight_decay=weight_decay
)
weight_scheduler = torch.optim.lr_scheduler.StepLR(optim, weight_step, gamma)
print("[RSA]Optimizer is built. ")


""" Loss
"""
# consistant loss
criterion_consistance = nn.L1Loss()
# triplet loss
cl_margin = config["RSA"]["training"]["cl_margin"]
criterion_triplet = TripletLoss(cl_margin)


# =================================================================================== #
#                             4. Training                                             #
# =================================================================================== #
""" Training
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
# lambda_3 = config["RSA"]["training"]["lambda_3"]

for iter in range(n_iterations):
    # get data
    orig_spk_emb, mel = next(train_iter)
    mel_ = mel[:, :, :180]
    mel = mel[:, :, 20:]
    orig_spk_emb = orig_spk_emb.to(torch.float32)
    mel = mel.to(torch.float32)
    mel_ = mel_.to(torch.float32)
    key = torch.randn(batch_size, key_len).to(torch.float32)
    # load to device
    orig_spk_emb = cc(orig_spk_emb)
    mel = cc(mel)
    key = cc(key)
    # get condition
    # print(key.shape, acg_cond.shape) # torch.Size([B, 192]) torch.Size([B, 1])
    cond, _ = acg.reverse_sample(key, acg_cond)
    cond = cond.squeeze()
    # forward & backward processes of cINN
    mel_ano = rsa(mel, cond)
    # mel_res = rsa(mel_ano, cond, True)
    mel_recon = rsa(mel, orig_spk_emb)
    # # test
    # if iter == 0:
    #     print("key.shape: ", key.shape)
    #     print("acg_cond.shape: ", acg_cond.shape)
    #     print("cond.shape: ", cond.shape)
    #     print("orig_spk_emb.shape: ", orig_spk_emb.shape)
    #     print("mel.shape: ", mel.shape)
    #     print("mel_ano.shape: ", mel_ano.shape)
    #     print("mel_res.shape: ", mel_res.shape)
    #     print("mel_recon.shape: ", mel_recon.shape)
    #     print("mel_ts.shape: ", mel_ts.shape)
    # calculate losses
    consistance_loss = criterion_consistance(mel, mel_recon)
    anchor_emb = d_vec_enc(mel.transpose(1, 2))
    positive_emb = d_vec_enc(mel_.transpose(1, 2))
    negative_emb = d_vec_enc(mel_ano.transpose(1, 2))
    triplet_loss = criterion_triplet(anchor_emb, positive_emb, negative_emb)
    # save losses
    consistance_loss_history.append(consistance_loss)
    triplet_loss_history.append(triplet_loss)
    # total loss
    total_loss = lambda_1 * consistance_loss + lambda_2 * triplet_loss
    # nn backward
    optim.zero_grad()
    total_loss.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(
        rsa.parameters(), max_norm=config["RSA"]["training"]["grad_norm"]
    )
    optim.step()
    weight_scheduler.step()
    # logging
    print(
        f"[RSA]:[{iter+1}/{n_iterations}]",
        f"loss_consistance={consistance_loss.item():6f}",
        f"loss_triplet={triplet_loss.item():6f}",
        end="\r",
    )
    # summary
    if (iter + 1) % summary_steps == 0 or iter + 1 == n_iterations:
        print()
    # autosave
    if (iter + 1) % autosave_steps == 0 or iter + 1 == n_iterations:
        torch.save(rsa.state_dict(), f"{save_path}/rsa.ckpt")
        torch.save(optim.state_dict(), f"{save_path}/optim.opt")
