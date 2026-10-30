import os
import random

import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    """Seed python, numpy and torch (CPU + CUDA). cudnn.benchmark stays on for speed; results are
    reproducible up to GPU non-determinism."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def worker_init(worker_id: int) -> None:
    s = torch.initial_seed() % 2**32
    np.random.seed(s)
    random.seed(s)
