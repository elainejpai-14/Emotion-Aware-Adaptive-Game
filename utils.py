# utils.py
import os
import random
import numpy as np
import torch
from collections import deque, Counter

def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # These two are safe even if you don't use cuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_device():
    # Prefer Apple Metal (MPS), then CUDA, then CPU
    if torch.backends.mps.is_available():
        return torch.device('mps')
    if torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')

class EmotionSmoother:
    """
    Keeps a rolling window of predictions and returns the mode
    to reduce jitter in real-time emotion classification.
    """
    def __init__(self, window=7):
        self.window = window
        self.deq = deque(maxlen=window)

    def update(self, label: str):
        self.deq.append(label)
        if not self.deq:
            return label
        counts = Counter(self.deq)
        return counts.most_common(1)[0][0]

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

