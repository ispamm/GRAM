import os
import sys
import json
import random
import argparse
import numpy as np
import torch.distributed as dist
from easydict import EasyDict as edict
from utils.logger import LOGGER
import os

def parse_with_config(parser):

    args = parser.parse_args()  
    file_cfg = edict(json.load(open(args.config)))


    cmd_cfg_keys = {arg[2:].split('=')[0] for arg in sys.argv[1:]
                        if arg.startswith('--')}

    ### load default run_cfg 
    run_cfg = edict(json.load(open(file_cfg.run_cfg.default)))
    ### overwrite run_cfg by config file 
    run_cfg.update(file_cfg.run_cfg)
    ### overwrite run_cfg by cmd
    for k in cmd_cfg_keys:
        if k in run_cfg:
            run_cfg[k] = getattr(args,k) 

    
    # if file_cfg['model_cfg']: must have

    ### load default model_cfg
    model_cfg = edict(json.load(open(file_cfg.model_cfg.default)))
    ### overwrite model_cfg by config file 
    model_cfg.update(file_cfg.model_cfg)
    


    if args.pretrain_dir:
        ### load pretrained model_cfg
        pretrain_model_cfg = edict(json.load(open(os.path.join(args.pretrain_dir,'log','hps.json')))).model_cfg
        ### overwite inherit_keys
        global_inherit_keys = ['vision_encoder_type','pool_video']
        inherit_keys = list(set(global_inherit_keys)|set(model_cfg.inherit_keys))
        inherit_model_cfg = edict({k:v for k,v in pretrain_model_cfg.items() if k in inherit_keys})
        model_cfg.update(inherit_model_cfg)
        
    # else:
    #     ### load from specific path
    #     assert args.model_cfg_file
    #     model_cfg = edict(json.load(open(args.model_cfg_file)))

    ### overwrite model_cfg by cmd
    for k in cmd_cfg_keys:
        if k in model_cfg:
            model_cfg[k] = getattr(args,k) 


    ### load data_cfg from config file 
    data_cfg = file_cfg['data_cfg']

    ### overwrite data_cfg by cmd, only valid when single dataset
    for k in cmd_cfg_keys:
        if k.startswith('train_'):
            assert len(data_cfg.train)==1 or k in ['train_batch_size','train_task']

            if k=='train_epoch':             
                data_cfg.train[0].epoch = args.train_epoch
            elif k=='train_steps':           
                data_cfg.train[0].steps = args.train_steps
            elif k=='train_vision_sample_num': 
                data_cfg.train[0].vision_sample_num = args.train_vision_sample_num
            elif k=='train_batch_size':  
                for i in range(len(data_cfg.train)):
                    data_cfg.train[i].batch_size = args.train_batch_size
            elif k=='train_task':   
                for i in range(len(data_cfg.train)):
                    data_cfg.train[i].task = args.train_task
        elif k.startswith('test'):
            # assert len(data_cfg.val)==1
            for i in range(len(data_cfg.val)):
                if k=='test_batch_size':         
                    data_cfg.val[i].batch_size = args.test_batch_size
                elif k=='test_vision_sample_num':
                    data_cfg.val[i].vision_sample_num = args.test_vision_sample_num
                elif k=='test_task':         
                    data_cfg.val[i].task = args.test_task

        elif k=='vision_transforms':         
            assert len(data_cfg.train)==1
            assert len(data_cfg.val)==1
            data_cfg.train[0]['vision_transforms'] = args.vision_transforms
            data_cfg.val[0]['vision_transforms'] = args.vision_transforms
        
        


    ### general configurations for different models, transmit directly from run_cfg

    # model_cfg.vision_resolution = run_cfg.vision_resolution
    # model_cfg.max_length = run_cfg.max_length
    # model_cfg.min_length = run_cfg.min_length
    # model_cfg.max_output_txt_len = run_cfg.max_output_txt_len
    # model_cfg.beam_size = run_cfg.beam_size
    # model_cfg.prompt = run_cfg.prompt
    # model_cfg.checkpointing = run_cfg.checkpointing
    # model_cfg.frozen_vision = run_cfg.frozen_vision
    # model_cfg.captioner_mode = run_cfg.captioner_mode
    # model_cfg.generate_nums = run_cfg.generate_nums


    ### special rules

    if model_cfg.checkpointing:
        run_cfg.use_ddp = False

    data_cfg.concatenated_nums = getattr(model_cfg,'concatenated_nums',1) ### for cosa training

    max_vision_sample_num = compute_max_vision_sample_num_for_position_embeddings(data_cfg)
    max_audio_sample_num = compute_max_audio_sample_num_for_position_embeddings(data_cfg)

    model_cfg.max_vision_sample_num = max_vision_sample_num
    model_cfg.max_audio_sample_num = max_audio_sample_num

    if run_cfg.bf16:
        run_cfg.fp16 = False 
    ### output cfg
 
    output_cfg = edict({'run_cfg':run_cfg,
                        'model_cfg':model_cfg, 
                        'data_cfg':data_cfg, 
                        'local_rank':args.local_rank})

    return output_cfg



    

def compute_max_vision_sample_num_for_position_embeddings(data_cfg):
    data_cfg_train = data_cfg.train
    vision_sample_num_ls_train=[]
    for d_cfg in data_cfg_train:
        vision_sample_num = d_cfg.get('vision_sample_num',1)
        vision_sample_num_ls_train.append(vision_sample_num * data_cfg.concatenated_nums)
        

    data_cfg_val = data_cfg.val
    vision_sample_num_ls_val=[]
    for d_cfg in data_cfg_val:
        vision_sample_num = d_cfg.get('vision_sample_num',1)
        vision_sample_num_ls_val.append(vision_sample_num )
       

    max_vision_sample_num = max(vision_sample_num_ls_train) if vision_sample_num_ls_train else max(vision_sample_num_ls_val)

    assert max_vision_sample_num  > 0
    return max_vision_sample_num

def compute_max_audio_sample_num_for_position_embeddings(data_cfg):
    data_cfg_train = data_cfg.train
    audio_sample_num_ls_train=[]
    for d_cfg in data_cfg_train:
        audio_sample_num = d_cfg.get('audio_sample_num',1)
        audio_sample_num_ls_train.append(audio_sample_num * data_cfg.concatenated_nums)
        

    data_cfg_val = data_cfg.val
    audio_sample_num_ls_val=[]
    for d_cfg in data_cfg_val:
        audio_sample_num = d_cfg.get('audio_sample_num',1)
        audio_sample_num_ls_val.append(audio_sample_num )
       

    max_audio_sample_num = max(audio_sample_num_ls_train) if audio_sample_num_ls_train else max(audio_sample_num_ls_val)

    assert max_audio_sample_num  > 0
    return max_audio_sample_num


def logging_cfgs(opts):
    with open(os.path.join(opts.run_cfg.output_dir, 'log', 'hps.json'), 'w') as writer:
        json.dump(vars(opts), writer, indent=4)

    n_gpu = dist.get_world_size()

    LOGGER.info('==='*6+'model_configs'+'==='*6+'\n')
    for k,v in opts.model_cfg.items():
        LOGGER.info(f'model_cfg_{k} : {v}')
    LOGGER.info('==='*6+'run_configs'+'==='*6+'\n')
    for k,v in opts.run_cfg.items():
        LOGGER.info(f'run_cfg_{k} : {v}')  
    LOGGER.info('==='*6+'data_configs'+'==='*6+'\n')
    for cfg in opts.data_cfg.train:
        name = cfg.name
        for k,v in cfg.items():
            LOGGER.info(f'data_cfg_{name}_train_{k} : {v}')  
    for cfg in opts.data_cfg.val:
        name = cfg.name
        for k,v in cfg.items():
            LOGGER.info(f'data_cfg_{name}_val_{k} : {v}')  

def str2bool(b):
    if b.lower() in ["false"]:
        return False
    elif b.lower() in ["true"]:
        return True
    elif b is None:
        return None
    else:
        raise Exception("Invalid Bool Value")


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vision_resolution", default=224, type=int)
    parser.add_argument("--local-rank", type=int, default=os.environ['LOCAL_RANK'])
    parser.add_argument("--checkpoint", default=None, type=str)
    parser.add_argument("--output_dir", default='output/', type=str)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1)
    parser.add_argument("--learning_rate", default=None, type=float)
    parser.add_argument("--clip_lr", default=5e-7, type=float)
    parser.add_argument("--clip_lr_text", default=5e-7, type=float)
    parser.add_argument("--optim", default='adam', choices=['adam', 'adamax', 'adamw'])
    parser.add_argument("--betas", default=[0.9, 0.98], nargs='+')
    parser.add_argument("--dropout", default=0.1, type=float)
    parser.add_argument("--weight_decay", default=0.01, type=float)
    parser.add_argument("--grad_norm", default=5.0, type=float)
    parser.add_argument("--warmup_ratio", default=0.1, type=float)
    parser.add_argument("--opt_model", default=None, type=str)
    parser.add_argument("--llm_model", default=None, type=str)
    parser.add_argument('--resume', action = 'store_true', help='use txt out')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--fp16', type=str2bool, default=True)
    parser.add_argument('--bf16', type=str2bool, default=False)
    parser.add_argument('--config')
    parser.add_argument('--zero_shot', action='store_true')
    parser.add_argument('--scheduler', type=str, default='warmup_linear')
    parser.add_argument("--max_generation_len", type=int, default=40)
    parser.add_argument("--max_length", type=int, default=30)
    parser.add_argument("--min_length", type=int, default=8)
    parser.add_argument("--max_output_txt_len", type=int, default=256)
    parser.add_argument("--amp", type=str, default='apex')
    parser.add_argument("--train_id", type=str, default='')
    parser.add_argument("--test_id", type=str, default='')
    parser.add_argument("--train_task", type=str, default='')
    parser.add_argument("--test_task", type=str, default='')
    parser.add_argument("--test_batch_size", type=int, default=-1)
    parser.add_argument("--max_text_tokens", type=int, default=40)
    parser.add_argument("--train_batch_size", type=int, default=-1)
    parser.add_argument("--checkpointing", type=str2bool, default=False)
    parser.add_argument("--frozen_vision", type=str2bool, default=False)
    parser.add_argument("--scst_finetuning", type=str2bool, default=False)
    parser.add_argument("--use_proposal_conv", type=str2bool, default=True)
    parser.add_argument("--ret_bidirection_evaluation", type=str2bool, default=False)
    parser.add_argument('--trainer_type', type=str, default="") 
    parser.add_argument("--itm_rerank_num", type=int, default=50)
    parser.add_argument("--itm_ratio", type=float, default=1.0)
    parser.add_argument("--save_best", type=str2bool, default=False)
    parser.add_argument("--train_epoch", type=float, default=-1)
    parser.add_argument("--contra_ratio", type=float, default=1.0)
    parser.add_argument("--train_steps", type=int, default=-1)
    parser.add_argument("--train_vision_sample_num", type=int, default=-1)
    parser.add_argument("--test_vision_sample_num", type=int, default=-1)
    parser.add_argument("--train_audio_sample_num", type=int, default=-1)
    parser.add_argument("--log_steps", type=int, default=-1)
    parser.add_argument("--test_audio_sample_num", type=int, default=-1)
    parser.add_argument("--concatenated_nums", type=int, default=1)
    parser.add_argument('--vision_encoder_type', type=str, default='clip_vit_base_16')
    parser.add_argument('--frame_embedding_type', type=str, default='')
    parser.add_argument('--loss_type', type=str, default='')
    parser.add_argument('--vision_transforms', type=str, default='none')
    parser.add_argument('--multimodal_encoder_type', type=str, default='bert_base_uncased')
    parser.add_argument('--num_train_steps', type=int, default=0)
    parser.add_argument('--huggingface_trainer', type=str2bool, default=False)
    parser.add_argument('--pretrain_dir', type=str, default="") 
    parser.add_argument('--deepspeed', type=str, default="") 
    parser.add_argument('--prompt', type=str, default=None)   
    parser.add_argument('--model_cfg_file', type=str, default="")  
    parser.add_argument('--llm_type', type=str, default="")                  
    parser.add_argument('--dual_softmax', type=str2bool, default=False)
    parser.add_argument('--pool_video', type=str2bool, default=False)
    parser.add_argument('--use_flash_attn', type=str2bool, default=False)
    parser.add_argument('--qformer_question', type=str2bool, default=False)
    parser.add_argument('--frozen_llm', type=str2bool, default=True)
    parser.add_argument('--use_deepspeed', type=str2bool, default=False)
    parser.add_argument('--captioner_mode', type=str2bool, default=False)
    parser.add_argument('--qformer_text_input', type=str2bool, default=True)
    parser.add_argument('--evaluate_ret_text', type=str2bool, default=False)
    parser.add_argument('--pool_vision', type=str2bool, default=False)
    parser.add_argument('--first_eval', type=str2bool, default=True)
    parser.add_argument('--vision_perceiver_query_num', type=int, default=-1)
    parser.add_argument('--remove_before_ckpt', type=str2bool, default=True)
    parser.add_argument('--dataset_mix_type', type=str, default='random')
    parser.add_argument('--valid_freq', type=int, default=10)
    parser.add_argument('--new_params_name', type=str, default=[], nargs='+') 
    parser.add_argument('--new_lr', type=float, default=0.0)  
    parser.add_argument('--beam_size', type=int, default=3)
    parser.add_argument('--generate_nums', type=int, default=1)
    parser.add_argument('--beam_size_qa', type=int, default=1)
    parser.add_argument('--contra_dim', type=int, default=512)
    parser.add_argument('--mode', type=str, default='training')
    parser.add_argument('--perceiver_mode', type=str, default='')
    parser.add_argument('--vision_cut_frames', type=int, default=-1)
   

    args = parse_with_config(parser)

    return args



    
