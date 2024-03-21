""" Process Audio Data
    * Read from .wav files
    * Get mel spectrograms
    * Get spk embeddings
    * Build Dataset
"""

import os
import pickle
import random
import sys
import torchaudio
import yaml

from utils import *

sys.path.append("..")
sys.path.append("/Users/pecholalee/Coding/RSA")
from speech_brain_proxy import EncoderClassifier

asv = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb")

if __name__ == "__main__":
    config_path = "./data/config.yaml"

    # Read from dataConfig, get hyper parameters for STFT
    with open(config_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    data_path = config["data_path"]
    out_path = config["out_path"]
    sample_rate = config["sample_rate"]
    audio_limit_len = config["audio_limit_len"]
    mel_segment_len = config["mel_segment_len"]
    sample_offset = config["sample_offset"]

    preemph = config["preemph"]
    n_mels = config["n_mels"]
    n_fft = config["n_fft"]
    hop_size = config["hop_size"]
    win_size = config["win_size"]
    f_min = config["f_min"]

    data = []  # save all audio signal

    # Read audio data
    audio_path_list = []  # save absolute paths of audio files
    for root_path, dirs, files in os.walk(data_path):
        for file in files:
            file_path = os.path.join(root_path, file)
            if file_path.split(".")[-1] == "wav":
                audio_path_list.append(file_path)
    random.shuffle(audio_path_list)
    print(f"[Dataset]got {len(audio_path_list)} audio files")

    for i, audio_path in enumerate(audio_path_list):
        # Read & Trim & Resample
        audio, _, _ = read_resample(
            audio_path=audio_path, sr=sample_rate, audio_limit_len=None
        )
        audio_t, _ = torchaudio.load(audio_path)
        spk_emb = asv.encode_batch(audio_t).squeeze().numpy()
        mel = get_mel(
            audio,
            preemph,
            sample_rate,
            n_mels,
            n_fft,
            hop_size,
            win_size,
            f_min,
        )  # mel shape: [T, n_mels]
        j = 0
        while True:
            if j + mel_segment_len > mel.shape[0]:
                break
            mel_segment = mel[j : j + mel_segment_len]
            data.append((spk_emb, mel_segment))
            j = j + sample_offset
        print(
            f"[Dataset]processed {i+1} audio file(s), got {len(data)} mel-sprectrogram sample(s)",
            end="\r",
        )
    print()

    # Dump Pickle
    with open(os.path.join(out_path, "audio_mel.pkl"), "wb") as f:
        pickle.dump(data, f)
        print(f"[Dataset]dumped pickle to {os.path.join(out_path, 'audio_mel.pkl')}")
