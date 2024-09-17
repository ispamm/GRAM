import os
import random
import torch
import numpy as np
import torch.distributed as dist
from .logger import LOGGER,add_log_to_file

def initialize(opts):
    
    # if not os.path.exists(opts.run_cfg.output_dir):
    os.makedirs(os.path.join(opts.run_cfg.output_dir, 'log'), exist_ok=True)
    os.makedirs(os.path.join(opts.run_cfg.output_dir, 'ckpt'), exist_ok=True)
    
    local_rank = opts.local_rank
    print(local_rank)
    torch.cuda.set_device(local_rank)
    print("DEVICE SET")
    dist.init_process_group(backend='nccl') 
    if opts.run_cfg.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, "
                         "should be >= 1".format(
                            opts.run_cfg.gradient_accumulation_steps))
    set_random_seed(opts.run_cfg.seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True
    if dist.get_rank() == 0:
        # TB_LOGGER.create(os.path.join(opts.output_dir, 'log'))
        add_log_to_file(os.path.join(opts.run_cfg.output_dir, 'log', 'log.txt'))
    else:
        LOGGER.disabled = True


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

