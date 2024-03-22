""" 1st stage training: Anonymization Condition Generator
"""

import torch
import torch.optim
import yaml

from models.ACG import ACG
from data.dataset import SAdataset, get_data_loader, infinite_iter
from speech_brain_proxy import EncoderClassifier

config_path = "./models/config.yaml"
ckpt_path = "../SpkAno/RSA_data/save/acg.ckpt"
pickle_path = "../SpkAno/miniSAdata_pickle/acg_audio.pkl"

# _____________________________

# Build Model
acg = ACG(config_path=config_path)
print("[RSA](stage 1): ACG is built. ")
print(
    "[RSA](stage 1): total parameter count: {}".format(
        sum(x.numel() for x in acg.parameters())
    )
)
asv = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb")
print("[RSA](stage 1): ASV from SpeechBrain is built. ")

# Setup
with open(config_path) as f:
    config = yaml.load(f, Loader=yaml.FullLoader)
n_iterations = config["ACG"]["training"]["n_iterations"]
summary_steps = config["ACG"]["training"]["summary_steps"]
autosave_steps = config["ACG"]["training"]["autosave_steps"]
batch_size = config["ACG"]["training"]["batch_size"]
num_workers = config["ACG"]["training"]["num_workers"]

scheduler = torch.optim.lr_scheduler.MultiStepLR(
    acg.optim,
    milestones=[int(n_iterations * 0.6), int(n_iterations * 0.8)],
    gamma=0.1,
)

nll_mean = []


# Data
dataset = SAdataset(pickle_path)
train_iter = infinite_iter(
    get_data_loader(dataset=dataset, batch_size=batch_size, num_workers=num_workers)
)
print("[RSA](stage 1): DataLoader is built. ")


# Training
for i in range(n_iterations):
    """data"""
    data = next(train_iter).to(torch.float32)  # .cuda()
    input_data = asv.encode_batch(data)  # get x-vector
    cond = cond = torch.tensor(
        [
            [
                0,
            ]
        ]
    )  # static condition (i.e. no condition)

    """forward"""
    z, log_j = acg(input_data, cond)
    # print("input_data.shape: ", input_data.shape)
    # print("cond shape: ", cond.shape)
    # print("z.shape: ", z.shape)
    # print("log_j.shape: ", log_j.shape)
    # print("log_j: ", log_j)

    """loss"""
    nll = torch.mean(z**2) / 2 - torch.mean(log_j) / acg.ndim_total

    """backward"""
    nll.backward()
    torch.nn.utils.clip_grad_norm_(acg.trainable_parameters, 10.0)
    acg.optim.step()
    acg.optim.zero_grad()

    """log"""
    nll_mean.append(nll.item())
    print(
        f"[RSA](stage 1): [{i}/{n_iterations}]",
        f"loss={nll.item():6f}",
        end="\r",
    )
    if (i + 1) % summary_steps == 0 or i + 1 == n_iterations:
        print()

    """auto save"""
    if (i + 1) % autosave_steps == 0 or i + 1 == n_iterations:
        torch.save(acg.state_dict(), ckpt_path)
