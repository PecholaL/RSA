""" Read audio data from file
    * Func1: Read & Resample
"""

import os
import soundfile
import librosa
import resampy
import numpy
from scipy.signal import lfilter

""" Get single-channel audio and resample to 16kHz
"""


def read_resample(audio_path, sr=16000, audio_limit_len=None):
    assert os.path.exists(audio_path)

    data, origin_sr = soundfile.read(audio_path)

    # trim
    data, _ = librosa.effects.trim(data)

    # resample
    if origin_sr != sr:
        data = resampy.resample(data, origin_sr, sr)

    # limit to setted length (unit: second)
    audio_len = audio_len_second(data, sr)
    if audio_limit_len is not None:
        assert len(data) > 0
        if audio_len < audio_limit_len:
            repeats = int(audio_limit_len / audio_len) + 1
            data = numpy.tile(data, repeats)
        data = data[0 : sr * audio_limit_len]

    return data, sr, audio_len


def audio_len_second(audio, sr):
    return 1.0 * len(audio) / sr


def get_mel(
    x: numpy.ndarray,
    preemph: float,
    sample_rate: int,
    n_mels: int,
    n_fft: int,
    hop_length: int,
    win_length: int,
    f_min: int,
) -> numpy.ndarray:
    x = lfilter([1, -preemph], [1], x)
    magnitude = numpy.abs(
        librosa.stft(y=x, n_fft=n_fft, hop_length=hop_length, win_length=win_length)
    )
    mel_fb = librosa.filters.mel(sr=sample_rate, n_fft=n_fft, n_mels=n_mels, fmin=f_min)
    mel_spec = numpy.dot(mel_fb, magnitude)
    log_mel_spec = numpy.log(mel_spec + 1e-9)
    return log_mel_spec.T  # shape(T, n_mels)
