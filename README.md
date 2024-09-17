




# [NIPS2023]VAST: A Vision-Audio-Subtitle-Text Omni-Modality Foundation Model and Dataset

<div align=center><img src=img/radar_compare_alldata_vast.png/ width="75%" height="75%"></div>



[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/vast-a-vision-audio-subtitle-text-omni-1/video-retrieval-on-activitynet)](https://paperswithcode.com/sota/video-retrieval-on-activitynet?p=vast-a-vision-audio-subtitle-text-omni-1)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/vast-a-vision-audio-subtitle-text-omni-1/text-to-audio-retrieval-on-audiocaps)](https://paperswithcode.com/sota/text-to-audio-retrieval-on-audiocaps?p=vast-a-vision-audio-subtitle-text-omni-1)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/vast-a-vision-audio-subtitle-text-omni-1/audio-captioning-on-audiocaps)](https://paperswithcode.com/sota/audio-captioning-on-audiocaps?p=vast-a-vision-audio-subtitle-text-omni-1)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/vast-a-vision-audio-subtitle-text-omni-1/text-to-audio-retrieval-on-clotho)](https://paperswithcode.com/sota/text-to-audio-retrieval-on-clotho?p=vast-a-vision-audio-subtitle-text-omni-1)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/vast-a-vision-audio-subtitle-text-omni-1/audio-captioning-on-clotho)](https://paperswithcode.com/sota/audio-captioning-on-clotho?p=vast-a-vision-audio-subtitle-text-omni-1)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/vast-a-vision-audio-subtitle-text-omni-1/image-captioning-on-coco-captions)](https://paperswithcode.com/sota/image-captioning-on-coco-captions?p=vast-a-vision-audio-subtitle-text-omni-1)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/vast-a-vision-audio-subtitle-text-omni-1/video-retrieval-on-didemo)](https://paperswithcode.com/sota/video-retrieval-on-didemo?p=vast-a-vision-audio-subtitle-text-omni-1)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/vast-a-vision-audio-subtitle-text-omni-1/zero-shot-video-retrieval-on-didemo)](https://paperswithcode.com/sota/zero-shot-video-retrieval-on-didemo?p=vast-a-vision-audio-subtitle-text-omni-1)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/vast-a-vision-audio-subtitle-text-omni-1/zero-shot-cross-modal-retrieval-on-flickr30k)](https://paperswithcode.com/sota/zero-shot-cross-modal-retrieval-on-flickr30k?p=vast-a-vision-audio-subtitle-text-omni-1)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/vast-a-vision-audio-subtitle-text-omni-1/video-retrieval-on-msr-vtt)](https://paperswithcode.com/sota/video-retrieval-on-msr-vtt?p=vast-a-vision-audio-subtitle-text-omni-1)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/vast-a-vision-audio-subtitle-text-omni-1/video-question-answering-on-msrvtt-qa)](https://paperswithcode.com/sota/video-question-answering-on-msrvtt-qa?p=vast-a-vision-audio-subtitle-text-omni-1)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/vast-a-vision-audio-subtitle-text-omni-1/audio-visual-question-answering-on-music-avqa)](https://paperswithcode.com/sota/audio-visual-question-answering-on-music-avqa?p=vast-a-vision-audio-subtitle-text-omni-1)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/vast-a-vision-audio-subtitle-text-omni-1/tgif-frame-on-tgif-qa)](https://paperswithcode.com/sota/tgif-frame-on-tgif-qa?p=vast-a-vision-audio-subtitle-text-omni-1)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/vast-a-vision-audio-subtitle-text-omni-1/video-captioning-on-tvc)](https://paperswithcode.com/sota/video-captioning-on-tvc?p=vast-a-vision-audio-subtitle-text-omni-1)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/vast-a-vision-audio-subtitle-text-omni-1/audio-visual-captioning-on-valor-32k)](https://paperswithcode.com/sota/audio-visual-captioning-on-valor-32k?p=vast-a-vision-audio-subtitle-text-omni-1)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/vast-a-vision-audio-subtitle-text-omni-1/video-retrieval-on-vatex)](https://paperswithcode.com/sota/video-retrieval-on-vatex?p=vast-a-vision-audio-subtitle-text-omni-1)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/vast-a-vision-audio-subtitle-text-omni-1/video-retrieval-on-youcook2)](https://paperswithcode.com/sota/video-retrieval-on-youcook2?p=vast-a-vision-audio-subtitle-text-omni-1)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/vast-a-vision-audio-subtitle-text-omni-1/video-captioning-on-youcook2)](https://paperswithcode.com/sota/video-captioning-on-youcook2?p=vast-a-vision-audio-subtitle-text-omni-1)

<div align=center><img src=img/VAST-model.jpg/></div>



## Building Environment
VAST is implemented based on Pytorch. We use Python-3.9 and Cuda-11.7. Other version could be also compatible. Other needed packages are listed in preinstall.sh.

```
conda create -n vast python=3.9
conda activate vast
sh preinstall.sh
```

## Download basic encoder's pretrained checkpoints
make a dir named pretrained_weights under the main work dir.

1.download evaclip weight:
```
wget -P pretrained_weights/clip/ https://huggingface.co/QuanSun/EVA-CLIP/resolve/main/EVA01_CLIP_g_14_psz14_s11B.pt
```
2.download beats weight from https://github.com/microsoft/unilm/tree/master/beats

3.download bert weight:
```
from transformers import BertModel, BertTokenizer
bert = BertModel.from_pretrained('bert-base-uncased')
bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert.save_pretrained('pretrained_weights/bert/bert-base-uncased')
bert_tokenizer.save_pretrained('pretrained_weights/bert/bert-base-uncased')
```


The processed  pretrained_weights path should be as follows:
```
    ├── pretrained_weights
    │   ├── beats
    │   │   └── BEATs_iter3_plus_AS2M.pt
    │   ├── bert
    │   │   └── bert-base-uncased
    │   ├── clip
    │   │   └── EVA01_CLIP_g_14_psz14_s11B.pt
```

## Download  VAST models  and captioners (for labeling your own data)

make a dir named output under the main work dir.

1.download vast model (optional, for finetuning)

[[Google Drive Link](https://drive.google.com/file/d/1ZkeZpis2Fggy4MyTFPqQj37MgPZxdJ53/view?usp=sharing)]
[[Baidu Cloud Link](https://pan.baidu.com/s/18EYFsPGlqU1JdK_OIjLuow?pwd=v70h)]

2.vision captioner (optional, for labeling images/videos)

[[Google Drive Link](https://drive.google.com/file/d/1b4C8KzwYaJYytjRyBfEd1UVZNDjUc53_/view?usp=sharing)]
[[Baidu Cloud Link](https://pan.baidu.com/s/1QRLqHE8iiIYa4Ahf1bFQUA?pwd=vlkp)]

3.audio captioner  (optional, for labeling audios)

[[Google Drive Link](https://drive.google.com/file/d/1lv_tuv1_K-EBfutl14D1Hcc40t37pPhR/view?usp=sharing)]
[[Baidu Cloud Link](https://pan.baidu.com/s/1L_MOevCxDIyrGa1NnoqFhQ?pwd=1v2n)]

The processed  output path should be as follows:
```
    ├── output
    │   ├── vast
    │   │   ├── pretrain_vast
    │   │   ├── vision_captioner
    │   │   └── audio_captioner

```

## Download  VAST-27M annotations for pretraining

[[Google Drive Link](https://drive.google.com/drive/folders/14Y6S9hGm-YbkA8VlCgw4xxEB2fpCAURT?usp=sharing)]
[[Baidu Cloud Link](https://pan.baidu.com/s/1Zn0R5vXdrVr1jN7gHxPXdQ?pwd=76fs)]

Raw videos could be downloaded from YouTube.
## Download  downstream datasets annotations for finetuning
make a dir named datasets under the main work dir.

[[Google Drive Link](https://drive.google.com/file/d/1bOLUbbnPTgUp_Nc0PgORKC-174CwgwPm/view?usp=sharing)]
[[Baidu Cloud Link](https://pan.baidu.com/s/1sMeX7LBSSI-ODOmq5opsag?pwd=wxht)]


The processed  datasets path should be as follows:
```
    ├── output
    │   ├── annotations
    │   │   ├── msrvtt
    │   │   ├── ...
    │   │   └── msvd
    │   ├── srcdata
    │   │   ├── msrvtt
    │   │   ├── ...
    │   │   └── msvd
```
srcdata (images/videos/audios) should be collected by yourself.

## Finetune  Model
- finetune retrieval tasks
```
sh scripts/vast/finetune_ret.sh
```
- finetune captioning tasks
```
sh scripts/vast/finetune_cap.sh
```
- finetune QA tasks
```
sh scripts/vast/finetune_qa.sh
```

## Pretrain Model
```
sh scripts/pretrain_vast.sh
```

## Test your finetuned Model
For example, if the cmd for finetuning retrieval model is as follows:

```
python3 -m torch.distributed.launch \
--nnodes 1 \
--node_rank 0 \
--nproc_per_node 8 \
--master_port 9834 \
./run.py \
--learning_rate 2e-5 \
--checkpointing true \
--first_eval true \
--save_best true \
--config ./config/vast/finetune_cfg/retrieval-msrvtt.json \
--pretrain_dir $output_dir \
--output_dir $output_dir/downstream/retrieval-msrvtt \
```

if you want to test model, just add following two rows to the cmd:
```
--mode 'testing' \
--checkpoint /PATH/TO/SAVED_CHECKPOINT.pt
```


## Labeling your own data use vast's captioner
You need to prepare 1)a folder containing all videos/images or audios.

2)a meta.json composed of [{'video_id':'09WssDay9FE_1'},{'video_id':'09WssDay9FE_2'},...]

and then write the config file.
```
sh scripts/vast/vision_captioner.sh
sh scripts/vast/audio_captioner.sh
```


## Statement of common controllable items in cmd which can overwrite config files.
--train_vision_sample_num

--test_vision_sample_num

--train_audio_sample_num

--test_audio_sample_num

--train_task

--test_task

--learning_rate

--train_batch_size

--test_batch_size

--train_epoch 

--train_steps

--checkpointing

--frozen_vision

--valid_freq

--beam_size




## Citation

If you find this code useful for your research, please consider citing:


```
@article{chen2024vast,
  title={Vast: A vision-audio-subtitle-text omni-modality foundation model and dataset},
  author={Chen, Sihan and Li, Handong and Wang, Qunbo and Zhao, Zijia and Sun, Mingzhen and Zhu, Xinxin and Liu, Jing},
  journal={Advances in Neural Information Processing Systems},
  volume={36},
  year={2024}
}
```


## License

This project is released under the [MIT license](LICENSE)


## Third-Party Licenses

For the full list of third-party licenses used in this project, please see the [THIRD_PARTY_LICENSES.md](THIRD_PARTY_LICENSES.md) file.

