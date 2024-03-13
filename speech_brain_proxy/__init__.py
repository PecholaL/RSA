import sys
from pathlib import Path


BASE_DIR = "/Users/pecholalee/Coding/tools/speechbrain"

if Path(BASE_DIR).exists():
    sys.path.append(BASE_DIR)
else:
    sys.path.append("../../tools/speechbrain/speechbrain")

from speechbrain.inference.speaker import EncoderClassifier
