""" Anonymization process with trained RSA
"""

import json
import sys
import torch
import yaml
from scipy.io.wavfile import write
from scipy.io.wavfile import write

sys.path.append("/Users/pecholalee/Coding/RSA")
from hifigan.models import Generator
from hifigan.env import AttrDict
from models.RSA import RSA
from models.ACG import ACG


""" Paths
"""
vocoder_ckpt_path = "/Users/pecholalee/Coding/learning/Python/audioTest/generator_v3"
acg_ckpt_path = "/Users/pecholalee/Coding/SpkAno/RSA_data/save/acg.ckpt"
rsa_ckpt_path = "/Users/pecholalee/Coding/SpkAno/RSA_data/save/rsa.ckpt"
wav_path = "/Users/pecholalee/Coding/SpkAno/miniSAdataset/p225_003.wav"
out_path = "/Users/pecholalee/Coding/RSA/_TEST_ONLY_/out/ano.wav"


""" Build Hifi-GAN vocoder
"""
MAX_WAV_VALUE = 32768.0
device = "cpu"
with open("/Users/pecholalee/Coding/RSA/hifigan/config.json") as f:
    config = f.read()
json_config = json.loads(config)
h = AttrDict(json_config)
generator = Generator(h).to(device)
state_dict_g = torch.load(vocoder_ckpt_path, device)
generator.load_state_dict(state_dict_g["generator"])
generator.eval()
generator.remove_weight_norm()

""" Build RSA
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


# print(mel.shape)
# with torch.no_grad():
#     y_g_hat = generator(mel)
#     audio = y_g_hat.squeeze()
#     audio = audio * MAX_WAV_VALUE
#     audio = audio.cpu().numpy().astype("int16")
#     write("./_TEST_ONLY_/recon.wav", h.sampling_rate, audio)
