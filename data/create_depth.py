import sys
import os
from tqdm import tqdm
# Add the root directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch

from IndexAnno import AnnoIndexedDataset
from utils.args import get_args
import numpy as np
import random
args = get_args()
# initialize(args)

def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True
set_random_seed(args.run_cfg.seed)
data_cfg = args.data_cfg.train

for d_cfg in data_cfg:
    name = d_cfg['name']
    dataset = AnnoIndexedDataset(d_cfg, args)
    print(f"Dataset {name} loaded with {len(dataset)} samples")
    
    for i in tqdm(range(10000,160000)):
        if i >= len(dataset):
            break
        sample = dataset[i]