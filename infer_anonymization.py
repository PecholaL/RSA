""" Anonymization process with trained RSA
"""

import yaml
import sys
import torch
import json
from scipy.io.wavfile import write
from scipy.io.wavfile import write

sys.path.append("/Users/pecholalee/Coding/RSA")

from hifigan.models import Generator
from hifigan.env import AttrDict

################################################################################


MAX_WAV_VALUE = 32768.0

device = "cpu"

with open("/Users/pecholalee/Coding/RSA/hifigan/config.json") as f:
    config = f.read()
json_config = json.loads(config)
h = AttrDict(json_config)

generator = Generator(h).to(device)
state_dict_g = torch.load(
    "/Users/pecholalee/Coding/learning/Python/audioTest/generator_v3", device
)
generator.load_state_dict(state_dict_g["generator"])
generator.eval()
generator.remove_weight_norm()

mel = d[0][:][:].squeeze().T
print(mel.shape)
with torch.no_grad():
    y_g_hat = generator(mel)
    audio = y_g_hat.squeeze()
    audio = audio * MAX_WAV_VALUE
    audio = audio.cpu().numpy().astype("int16")
    write("./_TEST_ONLY_/recon.wav", h.sampling_rate, audio)
