import os
import random
import torch
import torchaudio
from utils.logger import LOGGER
from utils.tool import split
import audioread, librosa


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
       


    def read(self, id_):

        wav_file = os.path.join(self.audio_dir, id_)
        
        if not os.path.exists(wav_file):
            wav_file = os.path.join(self.audio_dir, id_+'.wav')
        if not os.path.exists(wav_file):
            wav_file = wav_file.replace('wav','mp3')
        if not os.path.exists(wav_file):
            wav_file = wav_file.replace('mp3','mkv')
        if not os.path.exists(wav_file):
            print('not have audios', id_)
            return torch.zeros(self.sample_num, self.target_length, self.melbins)
        try:
            if self.audio_encoder_type == 'ast':
                
                waveform, sr = torchaudio.load(wav_file)

                waveform = waveform - waveform.mean()
                fbank = torchaudio.compliance.kaldi.fbank(waveform, htk_compat=True, sample_frequency=sr, use_energy=False,
                                                        window_type='hanning', num_mel_bins=self.melbins, dither=0.0, frame_shift=self.frame_shift)

                                       
            
            elif self.audio_encoder_type == 'beats':
                #print(wav_file)
                #waveform, sr = torchaudio.load(wav_file,format="mp3")
                #audio_file = audioread.audio_open(wav_file)
                waveform, sr = librosa.load(wav_file,sr=None)
                #print(type(waveform))
                #print(waveform.shape)
                waveform=torch.tensor(waveform)
                waveform= waveform.unsqueeze(0)
                #print(waveform.shape)
                if sr != 16000:
                    trans = torchaudio.transforms.Resample(sr, 16000)
                    waveform = trans(waveform)
            
                waveform = waveform * 2 ** 15
                fbank = torchaudio.compliance.kaldi.fbank(waveform, num_mel_bins=self.melbins, sample_frequency=16000, frame_length=25, frame_shift=10)

            else:
                raise NotImplementedError

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
        
        except Exception as e:
            print(e)
            return    
                
