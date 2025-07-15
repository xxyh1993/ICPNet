import torch
import numpy as np
import random
import os

import torch.backends
import torch.backends.cuda
import torch.backends.cudnn


def setup_seed(seed=1):
    g = torch.Generator()
    g.manual_seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
