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
from torchvision.transforms.transforms import *
import os

import random
import torch

import numpy as np

from torchvision.transforms.transforms import *

from utils.logger import LOGGER

import os

import cv2

import os
import random
import torch
import torchaudio
from utils.logger import LOGGER
from utils.tool import split
import audioread, librosa

import warnings
warnings.filterwarnings('ignore')


def str2bool(b):
    if b.lower() in ["false"]:
        return False
    elif b.lower() in ["true"]:
        return True
    elif b is None:
        return None
    else:
        raise Exception("Invalid Bool Value")
import argparse

def get_args(pretrain_dir=None):
    class Args:
        vision_resolution = 224
        local_rank = int(os.environ['LOCAL_RANK'])
        checkpoint = None
        output_dir = 'output/'
        gradient_accumulation_steps = 1
        learning_rate = None
        clip_lr = 5e-7
        clip_lr_text = 5e-7
        optim = 'adam'
        betas = [0.9, 0.98]
        dropout = 0.1
        weight_decay = 0.01
        grad_norm = 5.0
        warmup_ratio = 0.1
        opt_model = None
        llm_model = None
        resume = False
        seed = 42
        fp16 = True
        bf16 = False
        config = None
        zero_shot = False
        scheduler = 'warmup_linear'
        max_generation_len = 40
        max_length = 30
        min_length = 8
        max_output_txt_len = 256
        amp = 'apex'
        train_id = ''
        test_id = ''
        train_task = ''
        test_task = ''
        test_batch_size = -1
        max_text_tokens = 40
        train_batch_size = -1
        checkpointing = False
        frozen_vision = False
        scst_finetuning = False
        use_proposal_conv = True
        ret_bidirection_evaluation = False
        trainer_type = ""
        itm_rerank_num = 50
        itm_ratio = 1.0
        save_best = False
        train_epoch = -1
        contra_ratio = 1.0
        train_steps = -1
        train_vision_sample_num = -1
        test_vision_sample_num = -1
        train_audio_sample_num = -1
        log_steps = -1
        test_audio_sample_num = -1
        concatenated_nums = 1
        vision_encoder_type = 'clip_vit_base_16'
        frame_embedding_type = ''
        loss_type = ''
        vision_transforms = 'none'
        multimodal_encoder_type = 'bert_base_uncased'
        num_train_steps = 0
        huggingface_trainer = False
        pretrain_dir = ""
        deepspeed = ""
        prompt = None
        model_cfg_file = ""
        llm_type = ""
        dual_softmax = False
        pool_video = False
        use_flash_attn = False
        qformer_question = False
        frozen_llm = True
        use_deepspeed = False
        captioner_mode = False
        qformer_text_input = True
        evaluate_ret_text = False
        pool_vision = False
        first_eval = True
        vision_perceiver_query_num = -1
        remove_before_ckpt = True
        dataset_mix_type = 'random'
        valid_freq = 10
        new_params_name = []
        new_lr = 0.0
        beam_size = 3
        generate_nums = 1
        beam_size_qa = 1
        contra_dim = 512
        mode = 'training'
        perceiver_mode = ''
        vision_cut_frames = -1

    

    args = Args()
    #return args
    args.config = './config/gram/finetune_cfg/retrieval-youcook.json'
    args.pretrain_dir = pretrain_dir
    args = parse_with_config(args)
    args.run_cfg.pretrain_dir = pretrain_dir
    args.run_cfg.use_ddp = False
    return args

def parse_with_config(args):

    #args = parser.parse_args()  
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
        #("loading pretrained model_cfg")
        #print(args.pretrain_dir)
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


class VisionMapper(object):
    def __init__(self, d_cfg, args):

        self.vision = d_cfg.vision
        self.name = d_cfg.name
        self.training = d_cfg.training
        self.vision_format = d_cfg.vision_format
        ### for feat extraction
        self.dense_extraction = getattr(d_cfg,'dense_extraction',False)
        self.extract_fps = getattr(d_cfg,'extract_fps',None)
        self.frame_fps = getattr(d_cfg,'frame_fps',None)


        if self.vision_format.startswith('video'):        
            self.sample_num = d_cfg.vision_sample_num 
        
        
        self.resolution = args.model_cfg.vision_resolution

        if args.model_cfg.vision_encoder_type.startswith('clip') or args.model_cfg.vision_encoder_type.startswith('evaclip'):
            self.mean = [0.48145466, 0.4578275, 0.40821073] 
            self.std  = [0.26862954, 0.26130258, 0.27577711]
            LOGGER.info(f'{self.name} Using clip mean and std.')
        else:    
            self.mean = [0.485, 0.456, 0.406]
            self.std  = [0.229, 0.224, 0.225]
            LOGGER.info(f'{self.name} Using imagenet mean and std.')

        self.vision_transforms =  d_cfg.get('vision_transforms','none')
        if self.vision_transforms == 'none':
            if self.training:
                self.transforms = Compose([
                                                Resize((self.resolution,self.resolution)),
                                                Normalize(self.mean,self.std)])
            else:
                self.transforms = Compose([
                                                Resize((self.resolution,self.resolution)),
                                                Normalize(self.mean,self.std)])
        elif self.vision_transforms == 'crop_flip':
            if self.training:
                self.transforms = Compose([
                                                RandomResizedCrop(self.resolution, [0.8,1.0],[1.0,1.0]),
                                                RandomHorizontalFlip(),
                                                Normalize(self.mean,self.std)])
            else:
                self.transforms = Compose([
                                                Resize(self.resolution),
                                                CenterCrop(self.resolution),
                                                Normalize(self.mean,self.std)])
            
        else:
            raise NotImplementedError
        LOGGER.info(f'{self.name} transforms {self.vision_transforms}')
    



    def read(self, video_path):
            cap = cv2.VideoCapture(video_path)

            # Get the total number of frames in the video
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            sample_num = 2

            if total_frames<sample_num:
                cap.release()
                return None, None

            # Create a list of frame IDs
            frames_ids = list(range(total_frames))
            # Split the frames for sampling
            frames_splited = np.array_split(frames_ids, sample_num)
            # Select the frames based on whether it's training or evaluation mode
            if self.training:
                sample_idx = [random.choice(i) for i in frames_splited]
            else:
                sample_idx = [i[(len(i) + 1) // 2 - 1] for i in frames_splited]
            # Extract the selected frames
            frames = []
            for idx in sample_idx:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)  # Set the video position to the selected frame
                ret, frame = cap.read()  # Read the frame
                if ret:
                    if frame.all()!=None: 
                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
                        frames.append(frame_rgb)
                    
                # else:
                #     return None #finisce qui
            
            while len(frames)<sample_num and frames_ids!=[]:
            # if len(frames)<sample_num:
                # print(f'problem with {id_}')
                # return None
                idx = random.choice(frames_ids) 
                frames_ids.remove(idx)
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)  # Set the video position to the selected frame
                ret, frame = cap.read()  # Read the frame
                if ret:
                   frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
                   frames.append(frame_rgb)
            if len(frames)<sample_num:
                cap.release()
                return None, None
            cap.release()  # Release the video capture object
            # Convert the list of frames to a NumPy array
            frames = np.array(frames) # (8, 480, 848, 3) oppure (8, 1080, 1920, 3) etc.
            # import pdb; pdb.set_trace()
            # depth_pixels = extract_depth(frames, save_path=video_path)
            
            
            # print(depth_pixels.shape)
            # Normalize and transpose the frames (N x H x W x C -> N x C x H x W)
            vision_pixels = torch.from_numpy(frames.transpose(0, 3, 1, 2) / 255.0)
            # pdb.set_trace()
            # Apply the necessary transforms
            vision_pixels = self.transforms(vision_pixels)
            return vision_pixels

class AudioMapper(object):
    # def __init__(self, audio_dir, opts, sample_num, check_exists=True):
    def __init__(self, d_cfg, args):
        self.audio_dir = d_cfg.audio
        self.melbins = args.model_cfg.audio_melbins
        self.target_length = args.model_cfg.audio_target_length
        self.training = d_cfg.training
        self.frame_shift = 10
        self.sample_num = d_cfg.audio_sample_num
        self.audio_encoder_type = args.model_cfg.audio_encoder_type
        if self.audio_encoder_type == 'ast':
            self.mean = -4.2677393
            self.std = 4.5689974
        elif self.audio_encoder_type == 'beats':
            self.mean =  15.41663
            self.std = 6.55582 
        else:
            raise NotImplementedError
       


    def read(self, audio_path):


        waveform, sr = librosa.load(audio_path,sr=None)

        waveform=torch.tensor(waveform)
        waveform= waveform.unsqueeze(0)

        if sr != 16000:
            trans = torchaudio.transforms.Resample(sr, 16000)
            waveform = trans(waveform)
            
        waveform = waveform * 2 ** 15
        fbank = torchaudio.compliance.kaldi.fbank(waveform, num_mel_bins=self.melbins, sample_frequency=16000, frame_length=25, frame_shift=10)



        # ### normalization
        fbank = (fbank - self.mean) / (self.std * 2)
        src_length = fbank.shape[0]
        # #### sample 
        output_slices = []
        pad_len = max(self.target_length * self.sample_num -src_length, self.target_length - src_length%self.target_length)
        fbank = torch.nn.ZeroPad2d((0, 0, 0, pad_len))(fbank)
        total_slice_num = fbank.shape[0] // self.target_length
        total_slice_num = list(range(total_slice_num))
        total_slice_num = split(total_slice_num, self.sample_num)
        
        if self.training:
            sample_idx = [random.choice(i) for i in total_slice_num]
        else:
            sample_idx = [i[(len(i)+1)//2-1] for i in total_slice_num]
        
        for i in sample_idx:
            cur_bank = fbank[i*self.target_length : (i+1)*self.target_length]
            output_slices.append(cur_bank)
        fbank = torch.stack(output_slices,dim=0)   ### n, 1024, 128
        return fbank
        



def build_batch(args, text,video,audio):
    assert len(text) == len(video) == len(audio)
    id_txt = [i for i in range(len(text))]
    id_ = [i for i in range(len(text))]
    raw_captions = text


    visionMapper = VisionMapper(args.data_cfg.train[0],args)
    audioMapper = AudioMapper(args.data_cfg.train[0],args)


    vision_pixels = []
    for i in range(len(video)):
        video_frames = visionMapper.read(video[i])
        if video_frames is not None:
            vision_pixels.append(video_frames)
    if vision_pixels == []:
        return None

    vision_pixels = torch.stack(vision_pixels).float()

    audio_spectrograms = []
    for i in range(len(audio)):
        #print(audio[i])
        audio_frames = audioMapper.read(audio[i])
        if audio_frames is not None:
            audio_spectrograms.append(audio_frames)
    if audio_spectrograms == []:
        return None
    audio_spectrograms = torch.stack(audio_spectrograms).float()

    
    batch = {}

    batch['ids'] = id_
    batch['raw_captions'] = raw_captions
    batch['vision_pixels'] = vision_pixels.cuda()
    batch['ids_txt'] = id_txt
    batch['audio_spectrograms'] = audio_spectrograms.cuda()



    return batch
