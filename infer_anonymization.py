""" Anonymization process with trained RSA
"""

import json
import sys
import torch
import warnings
import yaml
from scipy.io.wavfile import write

sys.path.append("/Users/pecholalee/Coding/RSA")
warnings.filterwarnings("ignore")

from hifigan.models import Generator
from hifigan.env import AttrDict
from data.utils import *
from models.RSA import RSA
from models.ACG import ACG


""" Paths
"""
vocoder_ckpt_path = "/Users/pecholalee/Coding/learning/Python/audioTest/generator_v3"
acg_ckpt_path = "/Users/pecholalee/Coding/SpkAno/RSA_data/save/acg.ckpt"
rsa_ckpt_path = "/Users/pecholalee/Coding/SpkAno/RSA_data/save/rsa.ckpt"
wav_path = "/Users/pecholalee/Coding/SpkAno/miniSAdataset/"
out_path = "/Users/pecholalee/Coding/RSA/_TEST_ONLY_/out/ano"

orig_wav_filename = "p225_003.wav"


""" Build Hifi-GAN vocoder
"""
MAX_WAV_VALUE = 32768.0
device = "cpu"
with open("./hifigan/config.json") as f:
    config = f.read()
json_config = json.loads(config)
h = AttrDict(json_config)
generator = Generator(h).to(device)
state_dict_g = torch.load(vocoder_ckpt_path, device)
generator.load_state_dict(state_dict_g["generator"])
generator.eval()
generator.remove_weight_norm()


""" Build & Load RSA
"""
config_path = "./models/config.yaml"
with open(config_path) as f:
    config = yaml.load(f, Loader=yaml.FullLoader)
acg = ACG(config)
rsa = RSA(config)
acg.load_state_dict(torch.load(acg_ckpt_path))
rsa.load_state_dict(torch.load(rsa_ckpt_path))


""" Read from .wav
"""
wav, _, _ = read_resample(wav_path + orig_wav_filename)
mel = get_mel(
    wav,
    preemph=0.98,
    sample_rate=16000,
    n_mels=80,
    n_fft=1024,
    hop_length=256,
    win_length=1024,
    f_min=0,
).T
mel = torch.from_numpy(mel).unsqueeze(0)
print("mel.shape", mel.shape)


""" Generate condition
"""
key = torch.randn(1, 192)
acg_cond = torch.tensor(
    [
        [
            0,
        ]
    ]
)
acg_cond = acg_cond.repeat(1, 1)
cond, _ = acg.reverse_sample(key, acg_cond)
cond = cond.squeeze(1, 2)
print("cond.shape", cond.shape)


""" Anonymization & Restoration
"""
mel_ano = rsa(mel, cond)
mel_res = rsa(mel_ano, cond, True)
print("mel_ano.shape", mel_ano.shape)
print("mel_res.shape", mel_res.shape)


""" Convert mel to wav
"""
with torch.no_grad():
    wav_ano = generator(mel_ano)
    audio = wav_ano.squeeze()
    audio = audio * MAX_WAV_VALUE
    audio = audio.cpu().numpy().astype("int16")
    write(out_path + orig_wav_filename, h.sampling_rate, audio)
print("[RSA]anonymization completed, saved to " + out_path + orig_wav_filename)
