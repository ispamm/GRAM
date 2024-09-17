import math
import torch
import random
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from easydict import EasyDict as edict
from torch.nn import LayerNorm as LayerNorm
from utils.logger import LOGGER

def gelu(x):
    """Implementation of the gelu activation function.
        For information: OpenAI GPT's gelu is slightly different (and gives slightly different results):
        0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
        Also see https://arxiv.org/abs/1606.08415
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))
    
class GELU(nn.Module):
    def forward(self, input_):
        output = gelu(input_)
        return output



class Contra_head(nn.Module):
    def __init__(self, input_dim, contra_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, contra_dim, bias=False)
    def forward(self, cls_token):
        return self.linear(cls_token)


class Match_head(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.linear1 = nn.Linear(hidden_size, hidden_size)
        self.activation = GELU()
        self.layernorm = LayerNorm(hidden_size, eps=1e-12)
        self.linear2 = nn.Linear(hidden_size, 2)
    def forward(self, cls_token):
        return self.linear2(self.layernorm(self.activation(self.linear1(cls_token))))


def disabled_train(self, mode=True):
    """Overwrite model.train with this function to make sure train/eval mode
    does not change anymore."""
    return self



class TokenMasker(nn.Module):
    def __init__(self, mask_token = -1, range_start=-1, range_end=-1):
        super().__init__()
        self.mask_token = mask_token
        self.range = [range_start,range_end]

    def forward(self, tokens, mask_prob):
        tokens = tokens.clone() ### important, must have
        tokens, labels = self.perform_mask(tokens, mask_prob)
        return tokens, labels

    
    def perform_mask(self, tokens, mask_prob):
        
        tokens = np.array(tokens.cpu().numpy())

        ### generate indicator first:
        mask_indicator = np.zeros(tokens.shape, dtype=np.int64)
        for i in range(len(mask_indicator)):
            while all(mask_indicator[i] == 0):
                for j in range(1, len(mask_indicator[0])):
                    if tokens[i][j]!=0 and random.random() < mask_prob:
                        mask_indicator[i][j] = 1
        
        


        labels = -np.ones(tokens.shape, dtype=np.int64) * 100 ### -100 ignore idx for nn.CrossEntropyLoss used in BERT
        for i in range(tokens.shape[0]):
            for j in range(tokens.shape[1]):
                
                if mask_indicator[i][j] == 1 :
                    src_token = tokens[i][j]
                    prob = random.random()   #### e-6 too much time
                    if prob < 0.8:
                        tokens[i][j] = self.mask_token  ### e-6 have no idea why too much 
                    elif prob < 0.9: 
                        tokens[i][j] = random.choice(list(range(*self.range)))   
                    #tokens[i][j] = self.mask_token
                    labels[i][j] = src_token


        tokens =torch.from_numpy(tokens).long().cuda()
        labels =torch.from_numpy(labels).long().cuda()
        
        return tokens, labels


   

class MMGeneralModule(nn.Module):
    def __init__(self):
        super().__init__()
      
    



    def modify_checkpoint(self, checkpoint):
        new_ckpt = {}
        for k,v in checkpoint.items():
            if 'video' in k:
                new_ckpt[k.replace('video','vision')]=v
            elif 'evaclip_model' in k:
                new_ckpt[k.replace('evaclip_model','vision_encoder')]=v
            elif 'clip_model' in k:    
                new_ckpt[k.replace('clip_model','vision_encoder')]=v
            else:
                new_ckpt[k] = v.float()
        
        checkpoint = new_ckpt

    
        if self.config.frame_embedding_type == 'adaptive':

            if 'vision_frame_embedding' in checkpoint:
                pretrain_embed = checkpoint['vision_frame_embedding']
                if pretrain_embed.shape[1]!=self.config.max_vision_sample_num:
                    pretrain_embed = F.interpolate(pretrain_embed.permute(0,2,1),self.config.max_vision_sample_num,mode='nearest').permute(0,2,1)
                    checkpoint['vision_frame_embedding'] = pretrain_embed

            else:
                    
                pretrain_embed = checkpoint['vision_perceiver.vision_frame_embedding']
                if pretrain_embed.shape[1]!=self.config.max_vision_sample_num:
                    pretrain_embed = F.interpolate(pretrain_embed.permute(0,2,1),self.config.max_vision_sample_num,mode='nearest').permute(0,2,1)
                    checkpoint['vision_perceiver.vision_frame_embedding'] = pretrain_embed

            if 'audio_frame_embedding' in checkpoint:
                pretrain_embed_a = checkpoint['audio_frame_embedding']
                if pretrain_embed_a.shape[1]!=self.config.max_audio_sample_num:
                    pretrain_embed_a = F.interpolate(pretrain_embed_a.permute(0,2,1),self.config.max_audio_sample_num,mode='nearest').permute(0,2,1)
                    checkpoint['audio_frame_embedding'] = pretrain_embed_a

        # if self.config.vision_resolution != pretrain_cfg['vision_resolution']:
        if self.config.vision_encoder_type.startswith('clip'):
            vision_width = checkpoint["vision_encoder.visual.positional_embedding"].shape[1]
            vision_layers = len([k for k in checkpoint.keys() if k.startswith("visual.") and k.endswith(".attn.in_proj_weight")])
            vision_patch_size = checkpoint["vision_encoder.visual.conv1.weight"].shape[-1]
            
            grid_size = round((checkpoint["vision_encoder.visual.positional_embedding"].shape[0] - 1) ** 0.5)
       
            src  = checkpoint["vision_encoder.visual.positional_embedding"]
            src_cls = src[0:1]
            src_oth = src[1:]
            new_grid_size = self.config.vision_resolution // vision_patch_size
            if new_grid_size!=grid_size:
                src_oth = F.interpolate(src_oth.reshape(grid_size,grid_size,vision_width).permute(2,0,1).unsqueeze(0),(new_grid_size,new_grid_size),mode='bilinear')
                src_oth = src_oth[0].permute(1,2,0).reshape(-1,src.shape[-1])
                tgt = torch.cat((src_cls,src_oth),dim=0)
                checkpoint["vision_encoder.visual.positional_embedding"] = tgt

        elif self.config.vision_encoder_type.startswith('evaclip'):

            vision_width = checkpoint["vision_encoder.visual.pos_embed"].shape[2]
            vision_layers = len([k for k in checkpoint.keys() if k.startswith("visual.") and k.endswith(".attn.in_proj_weight")])

            vision_patch_size = checkpoint["vision_encoder.visual.patch_embed.proj.weight"].shape[-1]
            
            grid_size = round((checkpoint["vision_encoder.visual.pos_embed"].shape[1] - 1) ** 0.5)
     
            src  = checkpoint["vision_encoder.visual.pos_embed"][0]
            src_cls = src[0:1]
            src_oth = src[1:]
            new_grid_size = self.config.vision_resolution // vision_patch_size
            if new_grid_size!=grid_size:
                src_oth = F.interpolate(src_oth.reshape(grid_size,grid_size,vision_width).permute(2,0,1).unsqueeze(0),(new_grid_size,new_grid_size),mode='bilinear')
                src_oth = src_oth[0].permute(1,2,0).reshape(-1,src.shape[-1])
                tgt = torch.cat((src_cls,src_oth),dim=0)
                checkpoint["vision_encoder.visual.pos_embed"] = tgt.unsqueeze(0)



        else:
            pass
        

     
        return checkpoint


    def construct_vision_encoder(self):
        ##### construct vision encoder 
        if self.config.vision_encoder_type.startswith('clip') or self.config.vision_encoder_type.startswith('evaclip'):
            self.load_clip_model() 
        elif self.config.vision_encoder_type.startswith('swin'):
            self.load_swin_model()        
        elif self.config.vision_encoder_type.startswith('videoswin'):
            self.load_videoswin_model()          
        else:
            raise NotImplementedError
    
        if self.config.frozen_vision:
            for k,v in self.vision_encoder.named_parameters():
                v.requires_grad = False 

            self.vision_encoder = self.vision_encoder.eval()
            self.vision_encoder.train = disabled_train
    

    def construct_audio_encoder(self):
        if self.config.audio_encoder_type.startswith('ast'):
            self.load_ast_model()
        elif self.config.audio_encoder_type.startswith('beat'):
            self.load_beats_model()

        if self.config.frozen_audio:
            for k,v in self.audio_encoder.named_parameters():
                v.requires_grad = False 

            self.audio_encoder = self.audio_encoder.eval()
            self.audio_encoder.train = disabled_train
    



    

    def load_videoswin_model(self):
        from .vision_encoders.videoswin.videoswin import SwinTransformer3D
        assert self.config.vision_encoder_type == 'videoswin_base_k600_22k'
        time_stride = 1
        self.vision_encoder = SwinTransformer3D(time_stride = 1, embed_dim=128, num_heads=[4, 8, 16, 32],checkpointing=self.config.checkpointing)
        self.vision_dim = 1024 
        videoswin_weight = torch.load('./pretrained_weights/videoswin_base_k600_22k.pth',map_location='cpu')['state_dict']
        videoswin_weight = {k.replace('backbone.',''):v for k,v in videoswin_weight.items()}
        missing_keys, unexpected_keys = self.vision_encoder.load_state_dict(videoswin_weight,strict=False)
        del(videoswin_weight)
        LOGGER.info(f'missing_keys in videoswin: {missing_keys}')
        LOGGER.info(f'unexpected_keys in videoswin : {unexpected_keys}')



    def load_beats_model(self):

        from .audio_encoders.beats.beats import  BEATsConfig, BEATs
        checkpoint = torch.load('./pretrained_weights/beats/BEATs_iter3_plus_AS2M.pt')
        cfg = BEATsConfig(checkpoint['cfg'])

        self.audio_encoder = BEATs(cfg, checkpointing = self.config.checkpointing)
        self.audio_encoder.load_state_dict(checkpoint['model'])
        self.audio_dim = 768



    def load_ast_model(self):
        cfg = edict({'attention_dropout':0.1,
                                'hidden_act':"gelu",
                                'hidden_dropout':0.1,
                                'hidden_size':768,
                                'initializer_range':0.02,
                                'intermediate_size':3072,
                                'num_attention_heads':12,
                                'num_hidden_layers':12})
        cfg.checkpointing = self.config.checkpointing
        cfg.audio_melbins = self.config.audio_melbins
        cfg.audio_target_length = self.config.audio_target_length
        from  .audio_encoders.ast.ast import  TransformerEncoder, AudioEmbeddings  
        self.audio_embeddings = AudioEmbeddings(cfg)
        self.audio_encoder = TransformerEncoder(cfg, mode='prenorm')
        self.audio_dim = 768
       
        ast_weight = torch.load('./pretrained_weights/audioset_10_10_0.4593.pth',map_location='cpu')
        audio_weight = {}
        audio_weight['audio_embeddings.cls_token']  = ast_weight['module.v.cls_token'] 
        audio_weight['audio_embeddings.distill_token']  = ast_weight['module.v.dist_token'] 
        audio_weight['audio_embeddings.first_conv.weight'] =  ast_weight['module.v.patch_embed.proj.weight']  ### need to permute?
        audio_weight['audio_embeddings.first_conv.bias'] = ast_weight['module.v.patch_embed.proj.bias']
        pos_weight = ast_weight['module.v.pos_embed'][0]
        pos_weight_cls = pos_weight[0:1]
        pos_weight_oth = pos_weight[2:]   #### give up the distilled token
        pos_weight_oth = pos_weight_oth.reshape(12, 101,-1).permute(2,0,1).unsqueeze(0)
        tar_patch_num_height = cfg.audio_melbins // 16
        tar_patch_num_width = cfg.audio_target_length // 16
        pos_weight_oth = F.interpolate(pos_weight_oth, size = (tar_patch_num_height,tar_patch_num_width),mode='bilinear').squeeze().permute(1,2,0).reshape(-1,768)
        pos_weight_oth = torch.cat((pos_weight_cls,pos_weight_oth),dim=0)
        audio_weight['audio_embeddings.position_embeddings.weight'] = pos_weight_oth

        for  i in range(12):
            audio_weight['audio_encoder.layer.'+str(i)+'.attention.linears.0.weight'] = ast_weight['module.v.blocks.'+str(i)+'.attn.qkv.weight'][:768,:]
            audio_weight['audio_encoder.layer.'+str(i)+'.attention.linears.0.bias'] = ast_weight['module.v.blocks.'+str(i)+'.attn.qkv.bias'][:768]
            audio_weight['audio_encoder.layer.'+str(i)+'.attention.linears.1.weight'] = ast_weight['module.v.blocks.'+str(i)+'.attn.qkv.weight'][768:2*768,:]
            audio_weight['audio_encoder.layer.'+str(i)+'.attention.linears.1.bias'] = ast_weight['module.v.blocks.'+str(i)+'.attn.qkv.bias'][768:2*768]
            audio_weight['audio_encoder.layer.'+str(i)+'.attention.linears.2.weight'] = ast_weight['module.v.blocks.'+str(i)+'.attn.qkv.weight'][2*768:,:]
            audio_weight['audio_encoder.layer.'+str(i)+'.attention.linears.2.bias']  = ast_weight['module.v.blocks.'+str(i)+'.attn.qkv.bias'][2*768:]
            audio_weight['audio_encoder.layer.'+str(i)+'.attention.linears.3.weight']  = ast_weight['module.v.blocks.'+str(i)+'.attn.proj.weight']
            audio_weight['audio_encoder.layer.'+str(i)+'.attention.linears.3.bias'] = ast_weight['module.v.blocks.'+str(i)+'.attn.proj.bias']
            audio_weight['audio_encoder.layer.'+str(i)+'.ff_layer.linear1.weight']  = ast_weight['module.v.blocks.'+str(i)+'.mlp.fc1.weight']
            audio_weight['audio_encoder.layer.'+str(i)+'.ff_layer.linear1.bias']  = ast_weight['module.v.blocks.'+str(i)+'.mlp.fc1.bias']
            audio_weight['audio_encoder.layer.'+str(i)+'.ff_layer.linear2.weight']  = ast_weight['module.v.blocks.'+str(i)+'.mlp.fc2.weight']
            audio_weight['audio_encoder.layer.'+str(i)+'.ff_layer.linear2.bias']  = ast_weight['module.v.blocks.'+str(i)+'.mlp.fc2.bias']
            audio_weight['audio_encoder.layer.'+str(i)+'.layernorm1.weight']  = ast_weight['module.v.blocks.'+str(i)+'.norm1.weight']
            audio_weight['audio_encoder.layer.'+str(i)+'.layernorm1.bias']  = ast_weight['module.v.blocks.'+str(i)+'.norm1.bias']
            audio_weight['audio_encoder.layer.'+str(i)+'.layernorm2.weight']  = ast_weight['module.v.blocks.'+str(i)+'.norm2.weight']
            audio_weight['audio_encoder.layer.'+str(i)+'.layernorm2.bias'] = ast_weight['module.v.blocks.'+str(i)+'.norm2.bias']
        audio_weight['audio_encoder.last_layernorm.weight'] = ast_weight['module.v.norm.weight']
        audio_weight['audio_encoder.last_layernorm.bias'] = ast_weight['module.v.norm.bias']

        missing_keys, unexpected_keys = self.load_state_dict(audio_weight, strict=False)
        LOGGER.info(f'missing_keys in ast: {missing_keys}')
        LOGGER.info(f'unexpected_keys in ast: {unexpected_keys}')
        del(ast_weight)
        del(audio_weight)



         
   
    def load_clip_model(self):


        if self.config.vision_encoder_type.startswith('evaclip'):
            from .vision_encoders.evaclip import create_model
            if  self.config.vision_encoder_type == 'evaclip02_base':
                model_name = "EVA02-CLIP-B-16" 
                pretrained = "./pretrained_weights/clip/EVA02_CLIP_B_psz16_s8B.pt" 
                self.vision_dim = 768
                
            elif self.config.vision_encoder_type == 'evaclip02_base_self':
                model_name = "EVA02-CLIP-B-16" 
                pretrained = "./pretrained_weights/clip/EVA02_B_psz14to16.pt"
                self.vision_dim = 768

            elif self.config.vision_encoder_type == 'evaclip02_large':
                model_name = "EVA02-CLIP-L-14" 
                pretrained = "./pretrained_weights/clip/EVA02_CLIP_L_psz14_s4B.pt"
                self.vision_dim = 1024

            elif self.config.vision_encoder_type == 'evaclip02_bige':
                model_name = "EVA02-CLIP-bigE-14-plus" 
                pretrained = "./pretrained_weights/clip/EVA02_CLIP_E_psz14_plus_s9B.pt" 
                self.vision_dim = 1792

            elif self.config.vision_encoder_type == 'evaclip01_giant':
                model_name = "EVA01-CLIP-g-14" 
                pretrained = "./pretrained_weights/clip/EVA01_CLIP_g_14_psz14_s11B.pt"
                self.vision_dim = 1408
            

            self.vision_encoder = create_model(model_name, pretrained, force_custom_clip=True, image_size = self.config.vision_resolution)
            
            if self.config.checkpointing:
                self.vision_encoder.set_grad_checkpointing()
        
        else:
            ### openai clip

            from .vision_encoders.clip.clip import build_model
            from .vision_encoders.clip.clip import Transformer
            if  self.config.vision_encoder_type == 'clip_vit_base_16':
                clip_weight = torch.jit.load('./pretrained_weights/clip/ViT-B-16.pt', map_location='cpu')
                self.vision_dim = 768
            elif self.config.vision_encoder_type == 'clip_vit_large_14_336px':
                clip_weight = torch.jit.load('./pretrained_weights/clip/ViT-L-14-336px.pt', map_location='cpu')
                self.vision_dim = 1024
            elif self.config.vision_encoder_type == 'clip_vit_base_32':
                clip_weight = torch.jit.load('./pretrained_weights/clip/ViT-B-32.pt', map_location='cpu')
                self.vision_dim = 768
            clip_weight = clip_weight.state_dict()

            self.vision_encoder = build_model(clip_weight, self.config.vision_resolution, self.config.checkpointing).float()
            
  
            
    
    def forward_vision_encoder(self, vision_pixels):   ### b,n,3,h,w
        
        b,n,_,h,w = vision_pixels.shape
        
        if self.config.vision_encoder_type.startswith('clip') or self.config.vision_encoder_type.startswith('evaclip'):

            vision_output = self.vision_encoder.visual(vision_pixels.reshape(b*n,3,h,w), return_all_features=True)
            vision_output = vision_output.reshape(b,-1,*vision_output.shape[-2:])
  
        
        elif self.config.vision_encoder_type.startswith('swin'):
            vision_output = self.vision_encoder(vision_pixels.reshape(b*n,3,h,w))
            vision_output = vision_output.reshape(b,-1,*vision_output.shape[-2:])
    
        elif self.config.video_encoder_type.startswith('videoswin'):
            vision_output = self.vision_encoder(vision_pixels.transpose(1,2)) ### b,c,n,h,w
            vision_output = vision_output.permute(0, 2, 3, 4, 1)  ###b,n,h,w,c
            vision_output = vision_output.reshape(b,n,-1,vision_output.shape[-1])

        else:
            raise NotImplementedError()

        return vision_output  #### B , n , x ,C  n = self.frame_num





    def forward_audio_encoder(self, audio_spectrograms):        
        if self.config.audio_encoder_type.startswith('ast'): 
            audio_spectrograms = audio_spectrograms.permute(0,1,3,2) ### 1028,64 to 64,1028
            b,n,h,w, = audio_spectrograms.shape
            audio_spectrograms = audio_spectrograms.reshape(-1,*audio_spectrograms.shape[2:])
            audio_embeddings = self.audio_embeddings(audio_spectrograms) 
            audio_output,_ = self.audio_encoder(audio_embeddings)
            audio_output = audio_output.reshape(b,n,-1,audio_output.shape[-1])

        elif self.config.audio_encoder_type.startswith('beats'): 
            b,n,h,w, = audio_spectrograms.shape
            audio_spectrograms = audio_spectrograms.reshape(-1,*audio_spectrograms.shape[2:])
            audio_output = self.audio_encoder(audio_spectrograms)
            audio_output = audio_output.reshape(b,n,-1,audio_output.shape[-1])

        else:
            raise NotImplementedError()      
            
        return audio_output


    def pool_vision_for_contra(self, feature):  #feature b ,n ,x ,c
        #### always use frame_avg  for retrieval
        if self.config.vision_encoder_type.startswith('clip') or self.config.vision_encoder_type.startswith('evaclip'):
            feature = feature[:,:,0]
        elif self.config.vision_encoder_type.startswith('swin'):
            feature = feature.mean(dim=2)


        feature = torch.mean(feature, dim=1)
        return feature


    def pool_text_for_contra(self, feature):  #feature b ,n ,x, c
        return feature[:,0]
    
    def pool_audio_for_contra(self, feature):
        if self.config.audio_encoder_type.startswith('ast'):
            feature = feature[:,:,0]
        elif self.config.audio_encoder_type.startswith('beats'):
            feature = feature.mean(dim=2)
        else:
            raise NotImplementedError        
        feature = torch.mean(feature, dim=1)
        return feature  






    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, nn.Linear):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)





    def get_multimodal_forward_input_vision(self, vision_output):

        b,n,x,c = vision_output.shape
        # if self.config.pool_video:
        #     vision_output = vision_output[:,:,0]
            
        vision_output = self.hidden_trans_vision_multimodal(vision_output)  

        if self.config.frame_embedding_type == 'adaptive':
            if n!=self.vision_frame_embedding.shape[1]: #### testing and interpolate
                # dtype = self.vision_frame_embedding.dtype
                vision_frame_embedding = F.interpolate(self.vision_frame_embedding.float().permute(0,2,1),n,mode='nearest').permute(0,2,1).to(self.vision_frame_embedding)
            else:
                vision_frame_embedding = self.vision_frame_embedding


            vision_output =  vision_output + vision_frame_embedding.unsqueeze(-2)



        elif self.config.frame_embedding_type == 'none':
            pass


        vision_output =  vision_output.reshape(b,-1,self.multimodal_dim) 

        if hasattr(self,'vision_type_embeddings'): ### for three modality
            vision_output =  vision_output + self.vision_type_embeddings

        # vision_output = vision_output[:,:450]
        return vision_output

    def get_multimodal_forward_input_audio(self, audio_output):
        b,n,x,c = audio_output.shape
      
        if n!= self.audio_frame_embedding.shape[1]: #### testing and interpolate
            audio_frame_embedding = F.interpolate(self.audio_frame_embedding.permute(0,2,1),n,mode='nearest').permute(0,2,1)
        else:
            audio_frame_embedding = self.audio_frame_embedding
        audio_output = self.hidden_trans_audio_multimodal(audio_output)
        audio_output =  audio_output + audio_frame_embedding.unsqueeze(-2)
        audio_output = audio_output.reshape(b,-1,self.multimodal_dim)
        audio_output = audio_output + self.audio_type_embeddings                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          
        return audio_output


    def get_multimodal_forward_input_subtitle(self, subtitle_output):
        subtitle_output = self.hidden_trans_subtitle_multimodal(subtitle_output)
        subtitle_output = subtitle_output + self.subtitle_type_embeddings    
        return subtitle_output


    def load_swin_model(self):
        from .vision_encoders.swin.swin import SwinTransformer
        from .vision_encoders.swin.swin_config import get_config

        if self.config.vision_encoder_type.startswith('swin_base_22k_224'):
            swin_config = get_config('./pretrained_weights/swin/swin_base_patch4_window7_224_22k.yaml')
            swin_weight = torch.load('./pretrained_weights/swin/swin_base_patch4_window7_224_22k.pth', map_location='cpu')['model']
            self.vision_dim=1024
        elif self.config.vision_encoder_type.startswith('swin_large_22k_224'):
            swin_config = get_config('./pretrained_weights/swin/swin_large_patch4_window7_224_22k.yaml')
            swin_weight = torch.load('./pretrained_weights/swin/swin_large_patch4_window7_224_22k.pth', map_location='cpu')['model']
            self.vision_dim=1536

        model_type = swin_config.MODEL.TYPE
        # accelerate layernorm
        if swin_config.FUSED_LAYERNORM:
            try:
                import apex as amp
                layernorm = amp.normalization.LayerNorm
            except:
                layernorm = None
                print("To use LayerNorm, please install apex.")
        else:
            import torch.nn as nn
            layernorm = nn.LayerNorm
        
        self.vision_encoder = SwinTransformer(img_size=swin_config.DATA.IMG_SIZE,
                                patch_size=swin_config.MODEL.SWIN.PATCH_SIZE,
                                in_chans=swin_config.MODEL.SWIN.IN_CHANS,
                                num_classes=swin_config.MODEL.NUM_CLASSES,
                                embed_dim=swin_config.MODEL.SWIN.EMBED_DIM,
                                depths=swin_config.MODEL.SWIN.DEPTHS,
                                num_heads=swin_config.MODEL.SWIN.NUM_HEADS,
                                window_size=swin_config.MODEL.SWIN.WINDOW_SIZE,
                                mlp_ratio=swin_config.MODEL.SWIN.MLP_RATIO,
                                qkv_bias=swin_config.MODEL.SWIN.QKV_BIAS,
                                qk_scale=swin_config.MODEL.SWIN.QK_SCALE,
                                drop_rate=swin_config.MODEL.DROP_RATE,
                                drop_path_rate=swin_config.MODEL.DROP_PATH_RATE,
                                ape=swin_config.MODEL.SWIN.APE,
                                norm_layer=layernorm,
                                patch_norm=swin_config.MODEL.SWIN.PATCH_NORM,
                                use_checkpoint=swin_config.TRAIN.USE_CHECKPOINT,
                                fused_window_process=swin_config.FUSED_WINDOW_PROCESS)

        
        missing_keys, unexpected_keys = self.vision_encoder.load_state_dict(swin_weight,strict=False)

        del(swin_weight)
        #LOGGER.info(f'missing_keys in vision encoder: {missing_keys}')
        LOGGER.info(f'unexpected_keys in vision encoder: {unexpected_keys}')




 
