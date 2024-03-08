""" 1st stage training: Anonymization Condition Generator
"""

import torch
import torch.nn as nn
import torch.optim
import numpy

from models.ACG import ACG
