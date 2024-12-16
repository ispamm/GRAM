import os 
import json
import torch
import torch.distributed as dist
from utils.args import get_args,logging_cfgs
from utils.initialize import initialize
from utils.build_model import build_model
from utils.build_optimizer import build_optimizer 
from utils.build_dataloader import create_train_dataloaders, create_val_dataloaders
from utils.pipeline import train, test
import os
import wandb

def main():

    ### init 
    #print(os.environ['LOCAL_RANK'])
    args = get_args()
    initialize(args)

    ### logging cfgs
    logging_cfgs(args)  

    if dist.get_rank() == 0: 
        # start a new wandb run to track this script
        wandb.init(
        # set the wandb project where this run will be logged
        project ="Training_scratch_msrvtt",

        # track hyperparameters and run metadata
        config={
        "desc":f"Train_NOITM_{args.data_cfg.train[0]['name']}",
        "batch-size-train": args.data_cfg.train[0]['batch_size'],
        "batch-size-val": args.data_cfg.val[0]['batch_size'],
        "ngpus":4,
        "architecture": "GRAM",
        "dataset": args.data_cfg.train[0]['name'],
        "epochs": args.data_cfg.train[0]['epoch'],
        "name": args.run_cfg.mode+"_finetuning_" + args.data_cfg.train[0]['name'] + "_valFrame=" + str(args.data_cfg.val[0]['vision_sample_num']),  
        "val_frame": str(args.data_cfg.val[0]['vision_sample_num']),
        "train_frame": str(args.data_cfg.train[0]['vision_sample_num']),
        }
        )


    if args.run_cfg.mode == 'training':

        ### create datasets and dataloader
        train_loader = create_train_dataloaders(args)
        val_loaders = create_val_dataloaders(args)
        for name, loader in val_loaders.items():
            print(f"val_loader: {name} has {len(loader)} batches")

        ### build model and optimizer

        model, optimizer_ckpt, start_step = build_model(args)

        optimizer = build_optimizer(model, args, optimizer_ckpt)


        ### start evaluation
        if args.run_cfg.first_eval or args.run_cfg.zero_shot:
            test(model, val_loaders, args.run_cfg)                                 
            if args.run_cfg.zero_shot:
                return 

        ### start training


        train(model, optimizer, train_loader, val_loaders, args, start_step = start_step, verbose_time=False)

    elif args.run_cfg.mode == 'testing':
        ### build model
        model,_,_ = build_model(args)

        ### create datasets and dataloader
        val_loaders = create_val_dataloaders(args)
        print("TESTING MODE")
        ### start evaluation
        test(model, val_loaders, args.run_cfg)                                 

    else:
        raise NotImplementedError


if __name__ == "__main__":
    main()
