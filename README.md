# Gramian Multimodal Representation Learning and Alignment

<div align=center><img src=img/gram_method-compresso-1.png/ width="75%" height="75%"></div>






## Building Environment
GRAM is implemented based on Pytorch. We use Python-3.9 and Cuda-11.7. Other version could be also compatible. Other needed packages are listed in preinstall.sh.

```
conda create -n gram python=3.9
conda activate gram
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

## Download  GRAM models as a starting point of the pretraining stage  

make a dir named output under the main work dir.

1.download gram model (optional, for finetuning)

[[Google Drive Link]()]

)]

The processed  output path should be as follows:
```
    ├── output
    │   ├── gram
    │   │   ├── pretrain_gram

```

## Download  VAST-27M annotations for pretraining

[[Google Drive Link](https://drive.google.com/drive/folders/14Y6S9hGm-YbkA8VlCgw4xxEB2fpCAURT?usp=sharing)]
[[Baidu Cloud Link](https://pan.baidu.com/s/1Zn0R5vXdrVr1jN7gHxPXdQ?pwd=76fs)]

Raw videos could be downloaded from YouTube.

<!-- ## Download  downstream datasets annotations for finetuning
make a dir named datasets under the main work dir.

[[Google Drive Link](https://drive.google.com/file/d/1bOLUbbnPTgUp_Nc0PgORKC-174CwgwPm/view?usp=sharing)]
[[Baidu Cloud Link](https://pan.baidu.com/s/1sMeX7LBSSI-ODOmq5opsag?pwd=wxht)] -->


The processed  datasets path should be as follows:
```
    ├── datasets
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

## Finetune  Model on the 150k subset of VAST27M
Download annotations150k.json file subset from data available in openreview submission.
Reference it in scripts/gram/finetune_ret.sh and in config/gram/finetune_cfg/finetune-area.json
```
sh scripts/gram/finetune_ret.sh
```


## Finetune  Model on downstream datasets
Change configuration internally at scripts/gram/finetune_ret.sh and then run

```
sh scripts/gram/finetune_ret.sh
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
--config ./config/gram/finetune_cfg/retrieval-msrvtt.json \
--pretrain_dir $output_dir \
--output_dir $output_dir/downstream/retrieval-msrvtt \
```

if you want to test model, just add following two rows to the cmd:
```
--mode 'testing' \
--checkpoint /PATH/TO/SAVED_CHECKPOINT.pt
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




## Third-Party Licenses

For the full list of third-party licenses used in this project, please see the [THIRD_PARTY_LICENSES.md](THIRD_PARTY_LICENSES.md) file.

