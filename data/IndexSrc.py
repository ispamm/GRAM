# from utils.logger import LOGGER
import pickle
import webdataset as wds
from torchvision import transforms
from torchvision.transforms.transforms import *
import io
from PIL import Image
import random
import torch
from os.path import join
import json
from easydict import EasyDict as edict 
import string
import os
from toolz.sandbox import unzip
import numpy as np
import decord




def split(frame_name_lists, sample_num):
    if len(frame_name_lists) < sample_num:   ###padding with the last frame
        frame_name_lists += [frame_name_lists[-1]]*(sample_num - len(frame_name_lists))
    k, m = divmod(len(frame_name_lists), sample_num)
    return [frame_name_lists[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in list(range(sample_num))]



class ArgClass:

    def __init__(self,d_cfg, args):
        self.resolution = args.model_cfg.vision_resolution
        if args.model_cfg.vision_encoder_type.startswith('clip') or args.model_cfg.vision_encoder_type.startswith('evaclip'):
            self.mean = [0.48145466, 0.4578275, 0.40821073] 
            self.std  = [0.26862954, 0.26130258, 0.27577711]
        else:       
            self.mean = [0.485, 0.456, 0.406]
            self.std  = [0.229, 0.224, 0.225]

        self.vision_transforms =  d_cfg.get('vision_transforms','none')
        self.training = d_cfg.training
        self.vision_format = d_cfg.vision_format
        self.txt_format = d_cfg.txt_format
        if self.vision_format.startswith('video'):        
            self.sample_num = d_cfg.vision_sample_num 
        



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
        
        if self.txt_format == 'json':
            self.txt = json.load(open(d_cfg.txt))
        else:
            self.txt = getattr(d_cfg,'txt',None)

        from transformers import BertTokenizer


        self.tokenizer = BertTokenizer.from_pretrained('./pretrained_weights/bert/bert-base-uncased')
        

    def process(self, item):

        # import ipdb 
        # ipdb.set_trace()
        raw_captions = None 
        if len(item) == 2:
            src,  __key__ = item
        elif len(item) == 3:
            src,  raw_captions, __key__ = item
            raw_captions = raw_captions.decode()
        
        id_ = __key__ if not '/' in __key__ else __key__.split('/')[1]
        ### process vision

        if self.vision_format.startswith('image'):
            img = Image.open(io.BytesIO(src)).convert("RGB")
            img = np.array(img).transpose(2,0,1)/255.0
            img = torch.from_numpy(img) 
            img = self.transforms(img)   
            vision_pixels = img.unsqueeze(0)
        
        elif self.vision_format.startswith('video'):
            file_obj = io.BytesIO(src)
            container = decord.VideoReader(file_obj) 
            frames_ids = list(range(len(container)))
            frames_splited = split(frames_ids, self.sample_num)
            sample_idx = [random.choice(i) for i in frames_splited]
            frames = container.get_batch(sample_idx).asnumpy()
            vision_pixels = torch.from_numpy(frames.transpose(0,3,1,2)/255.0)  ### nX3xHxW
            vision_pixels = self.transforms(vision_pixels)    

        ### process text
        

   
      
        if self.txt_format == 'json':
            raw_captions = self.txt[id_]   
        elif self.txt_format == 'dir':
            if os.path.exists(os.path.join(self.txt,id_[:5]+'.json')):
               
                files = json.load(open(os.path.join(self.txt,id_[:5]+'.json')))
                if id_[:5]+'/'+id_ in files:
                    chosen_ls = files[id_[:5]+'/'+id_]
                    raw_captions = random.choice(chosen_ls) 

                elif id_ in files:
                    chosen_ls = files[id_]
                    raw_captions = random.choice(chosen_ls)                           


        text = self.tokenizer(raw_captions) #### for bad captions, raise error and continue


        return vision_pixels, raw_captions, id_


def warn_and_continue(exn):
    """Call in an exception handler to ignore any exception, issue a warning,
    and continue."""
    print(exn)
    return True


def SrcIndexedDataset(d_cfg, args):

    data = ArgClass(d_cfg, args)
    vision = d_cfg.vision  
    if vision.endswith('json'):
        vision = json.load(open(vision))
    elif vision.endswith('tar'):
        vision = [vision]
    else:
        vision = [os.path.join(d_cfg.vision,i) for i in os.listdir(vision) if i.endswith('.tar')]

    # dataset = wds.WebDataset(vision, shardshuffle=True, resampled=True).shuffle(1000).\
    #     to_tuple("jpg", "txt", "__key__", handler=warn_and_continue).map(data.process,handler=warn_and_continue)
    
 

    if d_cfg.vision_format.startswith('video'):
        suffix = 'mp4'
    elif d_cfg.vision_format.startswith('image'):
        suffix = 'jpg'
    

    if d_cfg.name in ['laion400m']:
        dataset = wds.WebDataset(vision, shardshuffle=True, resampled=True).shuffle(1000).\
            to_tuple(suffix, "txt", "__key__",handler=warn_and_continue).map(data.process,handler=warn_and_continue)


    else:
        dataset = wds.WebDataset(vision, shardshuffle=True, resampled=True).shuffle(1000).\
            to_tuple(suffix,  "__key__", handler=warn_and_continue).map(data.process,handler=warn_and_continue)


    dataset.collate_fn = srcindexedcollate
    dataset.worker_init_fn = None
    dataset.use_sampler = False

    return dataset






def srcindexedcollate(inputs):
    
    batch = {}
    all_data = map(list, unzip(inputs))
    keys = ['vision_pixels', 
            'raw_captions', 
            'ids'
           ]

    for key, data in zip(keys, all_data):
  
        if data[0] is None:
            continue 
        elif isinstance(data[0], torch.Tensor):
            batch[key] = torch.stack(data, dim=0).float()
      
        else:
            batch[key] = data

    
    return batch








if __name__=='__main__':


  



    args = edict({'model_cfg':{'vision_resolution':224,
            'vision_encoder_type':'clip'}})
    d_cfg = edict({"vision":"/public/datasets/laion400m/Laion400m/all_tar_path.json",
            "txt":"/public/chensihan/projects/VLPPP/output/split_caps_laion",
                "txt_format":"dir",
                "name":"laion400m",
                "training":False,
                "vision_format":"image"})       

  
    dataset = SrcIndexedDataset(d_cfg, args)

    for i in dataset:
        print(i)