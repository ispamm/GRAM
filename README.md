<h2 align="center"> <a href="https://openreview.net/forum?id=ftGnpZrW7P">[ICLR 2025] Gramian Multimodal Representation Learning and Alignment</a></h2>

<h3 align="center"><a href="https://ispamm.github.io/GRAM/"> Project page here 🚀</a></h3>

<div align=center><img src=assets/gram_method-compresso-1.png width="75%" height="75%"></div>

<h5 align="center"> If you like our project, please give us a star ⭐ on GitHub for latest update.  </h2>


<h5 align="center">
     
 
[![arXiv](https://img.shields.io/badge/Arxiv-2412.11959-b31b1b.svg?logo=arXiv)](https://arxiv.org/abs/2412.11959)
[![OpenReview Discussion](https://img.shields.io/badge/OpenReview-Discussion-B762C1)](https://openreview.net/forum?id=ftGnpZrW7P)

[![License](https://img.shields.io/badge/Code%20License-MIT-yellow)](https://github.com/PKU-YuanGroup/LanguageBind/blob/main/LICENSE)
[![Hits](https://hits.seeyoufarm.com/api/count/incr/badge.svg?url=https%3A%2F%2Fgithub.com%2Fispamm%2FGRAM&count_bg=%2379C83D&title_bg=%23555555&icon=&icon_color=%23E7E7E7&title=hits&edge_flat=false)](https://hits.seeyoufarm.com)
![GitHub Issues or Pull Requests](https://img.shields.io/github/issues/ispamm/GRAM)
![GitHub Issues or Pull Requests](https://img.shields.io/github/issues-closed/ispamm/GRAM)

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/gramian-multimodal-representation-learning/video-retrieval-on-msr-vtt)](https://paperswithcode.com/sota/video-retrieval-on-msr-vtt?p=gramian-multimodal-representation-learning)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/gramian-multimodal-representation-learning/zero-shot-video-retrieval-on-vatex)](https://paperswithcode.com/sota/zero-shot-video-retrieval-on-vatex?p=gramian-multimodal-representation-learning)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/gramian-multimodal-representation-learning/video-retrieval-on-vatex)](https://paperswithcode.com/sota/video-retrieval-on-vatex?p=gramian-multimodal-representation-learning)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/gramian-multimodal-representation-learning/zero-shot-video-retrieval-on-msr-vtt)](https://paperswithcode.com/sota/zero-shot-video-retrieval-on-msr-vtt?p=gramian-multimodal-representation-learning)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/gramian-multimodal-representation-learning/zero-shot-audio-retrieval-on-audiocaps)](https://paperswithcode.com/sota/zero-shot-audio-retrieval-on-audiocaps?p=gramian-multimodal-representation-learning)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/gramian-multimodal-representation-learning/zero-shot-video-retrieval-on-activitynet)](https://paperswithcode.com/sota/zero-shot-video-retrieval-on-activitynet?p=gramian-multimodal-representation-learning)

 <br>

</h5>

## 📰 News
* **[2025.04.17]** Takeaway functions available
* **[2025.01.22]** 🔥🔥🔥 Paper got accepted at ICLR 2025!! See you in Singapore!
* **[2024.12.18]** 🔥🔥🔥 The checkpoints are available [here](https://drive.google.com/drive/folders/15CGPSut2Bgcsuce1Fjaozfts0f9QK1Ya?usp=sharing)!
* **[2024.12.18]**  Code is available now! Welcome to **watch** 👀 this repository for the latest updates.
* **[2024.12.17]**  The paper has been published on Arxiv 🎉. The pdf version is available [here](https://arxiv.org/pdf/2412.11959)! 

## 😮 Highlights

### 💡 Radical change in the field of multimodal contrastive learning 
GRAM learns and then aligns modalities directly in the higher-dimensional space in which modality embeddings lie by minimizing the **Gramian volume of the k-dimensional parallelotope spanned by the modality vectors**, ensuring the geometric alignment of all modalities simultaneously.

### 🔥 SOTA Performance in almost all retrieval task
GRAM can replace cosine similarity in any downstream method, holding for 2 to modality and providing more meaningful alignment with respect to previous similarity measures. Moreover, the novel GRAM-based contrastive loss function enhances the alignment of multimodal models in the higher-dimensional embedding space, leading to new state-of-the-art performance in downstream tasks such as video-audio-text retrieval and audio-video classification.



### 👀 Multimodal alignment unlock new and fancy downstream task

An aligned shared latent space among n modalities is a strong baseline for whatever downstream task that rely on embedding extraction. The results obtained from this paper will lead to superior performance in existing downstream tasks (T2I, T2V, V2A, etc.) but also unlock fancy tasks such as for example image to audio generation or image generation conditioned on text and audio.

## 🚀 Main Results

<div align=center><img src=assets/results.png width="75%" height="75%"></div>

### ✨ Takeaway functions 

`simple_volume_computation`

This function computes the volume of the k-dimensional parallelotope formed by three vectors—one from each modality—using their Gram matrix determinant:

```python
def simple_volume_computation(language, video, audio):
    A = torch.stack([language, video, audio])
    G = A @ A.T
    gramian = torch.linalg.det(G)
    return torch.sqrt(gramian)
```

* A: Stacks the three modality vectors.

* G: Constructs the Gram matrix from dot products.

* det(G): Gives the squared volume of the parallelepiped formed by the vectors.

* sqrt(det(G)): Returns the actual volume.

This simple geometric operation scales to batches and more complex setups in the full GRAM function below.

`volume_computation`


```python
def volume_computation(anchor, *inputs):
    """
    General function to compute volume for contrastive learning loss functions.
    Compute the volume metric for each vector in anchor batch and all the other modalities listed in *inputs.

    Args:
    - anchor (torch.Tensor): Tensor of shape (batch_size1, dim)
    - *inputs (torch.Tensor): Variable number of tensors of shape (batch_size2, dim)

    Returns:
    - torch.Tensor: Tensor of shape (batch_size1, batch_size2) representing the volume for each pair.
    """
    batch_size1 = anchor.shape[0]
    batch_size2 = inputs[0].shape[0]

    # Compute pairwise dot products for language with itself
    aa = torch.einsum('bi,bi->b', anchor, anchor).unsqueeze(1).expand(-1, batch_size2)

    # Compute pairwise dot products for language with each input
    l_inputs = [anchor @ input.T for input in inputs]

    # Compute pairwise dot products for each input with themselves and with each other
    input_dot_products = []
    for i, input1 in enumerate(inputs):
        row = []
        for j, input2 in enumerate(inputs):
            dot_product = torch.einsum('bi,bi->b', input1, input2).unsqueeze(0).expand(batch_size1, -1)
            row.append(dot_product)
        input_dot_products.append(row)

    # Stack the results to form the Gram matrix for each pair
    G = torch.stack([
        torch.stack([aa] + l_inputs, dim=-1),
        *[torch.stack([l_inputs[i]] + input_dot_products[i], dim=-1) for i in range(len(inputs))]
    ], dim=-2)

    # Compute the determinant for each Gram matrix
    gram_det = torch.det(G.float())

    # Compute the square root of the absolute value of the determinants
    res = torch.sqrt(torch.abs(gram_det))
    return res
```

### 🧐 how to use it in practice?  Implementation of the InfoNCE loss with Volume:

```python
import torch
import torch.nn.functional as F

# Hyperparameters
bs = 32
latent_dim = 512
contrastive_temp = 0.07

# Output of the encoders
language = torch.randn((bs,latent_dim))
video = torch.randn((bs,latent_dim))
audio = torch.randn((bs,latent_dim))

volume = volume_computation(language,video,audio)
volume = volume / contrastive_temp


volumeT = volume_computation(language,video,audio).T
volumeT = volumeT / contrastive_temp

targets = torch.linspace(0, bs - 1, bs, dtype=int)

loss = (
        F.cross_entropy(-volume, targets, label_smoothing=0.1) #d2a
        + F.cross_entropy(-volumeT, targets, label_smoothing=0.1) #a2d
) / 2

print(loss)

```


## Building Environment
GRAM is implemented based on Pytorch. We use Python-3.9 and Cuda-11.7. Other version could be also compatible. Other needed packages are listed in preinstall.sh.

```
conda create -n gram python=3.9
conda activate gram
sh preinstall.sh
```

## Download basic encoder's pretrained checkpoints
Make a dir named pretrained_weights under the main work dir.

1. Download evaclip weight:
```
wget -P pretrained_weights/clip/ https://huggingface.co/QuanSun/EVA-CLIP/resolve/main/EVA01_CLIP_g_14_psz14_s11B.pt
```
2. Download beats weight from https://github.com/microsoft/unilm/tree/master/beats

3. Download bert weight:
```python
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

## MODEL ZOO

All models are available [here](https://drive.google.com/drive/folders/15CGPSut2Bgcsuce1Fjaozfts0f9QK1Ya?usp=sharing)! 

<div align="center">
<table border="1" width="100%">
    <tr align="center">
        <th>Name</th><th>Training Dataset</th><th>Testing Dataset</th><th>R@1 in Testing Dataset</th> <th>link</th>
    </tr>
    <tr align="center">
        <td>GRAM_pretrained_5modalities</td><td>Vast27M 150k Subset TVAS</td><td>MSRVTT</td><td>54.8</td><td><a href="https://drive.google.com/drive/folders/1T6IFymtJ8mKeO_amz9Kz1OFgHZEK6huz?usp=drive_link">link</a></td>
    </tr>
    <tr align="center">
        <td>GRAM_pretrained_4modalities</td><td>Vast27M 150k Subset TVASD</td><td>MSRVTT</td><td>55.3</td><td><a href="https://drive.google.com/drive/folders/1mD9PDvugLx3t1KtTCwJtYO8VZDFKfMgs?usp=drive_link">link</a></td>
    <tr align="center">
        <td>GRAM_finetuned_MSRVTT</td><td>MSRVTT</td><td>MSRVTT</td><td>64.0</td><td><a href="https://drive.google.com/drive/folders/1CEpdfL2xAm15FlYbmNd4r5c8fqTKDy16?usp=drive_link">link</a></td>
    </tr>
    <tr align="center">
        <td>GRAM_finetuned_DIDEMO</td><td>DIDEMO</td><td>DIDEMO</td><td>67.3</td><td><a href="https://drive.google.com/drive/folders/1tdiknBL-8yL06blKuLCn0uuxRe_Jpnee?usp=drive_link">link</a></td>
    </tr>
        <tr align="center">
        <td>GRAM_finetuned_ANET</td><td>ActivityNet</td><td>ActivityNet</td><td>69.9</td><td><a href="https://drive.google.com/drive/folders/1q4Q_qXb0IA0cmb-TO0WA6_89JiJf_apM?usp=sharing">link</a></td>
    </tr>
        </tr>
        <tr align="center">
        <td>GRAM_finetuned_VATEX</td><td>VATEX</td><td>VATEX</td><td>87.7</td><td><a href="https://drive.google.com/drive/folders/1klv4o7OSSPc1urGgdp3tRxgfU_XYKneW?usp=drive_link">link</a></td>
    </tr>

</table>
</div>


Download the entire folder that consists of a subfolder "log" and another one "ckpt. Place the folder whatever you prefer and record the location for future commands.

An example of paths after the download could be as follow:

```
    ├── pretrained_models
    │   ├── GRAM_pretrained_4modalities
    │   │   ├── log
    │   │   ├── ckpt    

```

## Inference for Multi-modal Binding

We have provided some sample datasets in [assets](https://github.com/ispamm/GRAM/tree/main/assets) to quickly see how languagebind works.

```python
from utils.utils_for_fast_inference import get_args, VisionMapper, AudioMapper, build_batch
from utils.build_model import build_model
from utils.volume import volume_computation3
import warnings
import os
warnings.filterwarnings("ignore") 


os.environ['LOCAL_RANK'] = '0'

#Pass the path to the pre-trained model folder
pretrain_dir = './gram_ckpt'

args = get_args(pretrain_dir)

model,_,_ = build_model(args)
model.to('cuda')

visionMapper = VisionMapper(args.data_cfg.train[0],args)
audioMapper = AudioMapper(args.data_cfg.train[0],args)



tasks = 'ret%tva'


text = ["A dog is barking","A dog is howling", "A red cat is meowing", "A black cat is meowing"]
video = ["./assets/videos/video1.mp4","./assets/videos/video2.mp4","assets/videos/video3.mp4","./assets/videos/video4.mp4"]
audio = ["./assets/audios/audio1.mp3","./assets/audios/audio2.mp3","./assets/audios/audio3.mp3","./assets/audios/audio4.mp3"]

batch = build_batch(args,text,video,audio)


evaluation_dict= model(batch, tasks, compute_loss=False)

feat_t = evaluation_dict['feat_t']
feat_v = evaluation_dict['feat_v']
feat_a = evaluation_dict['feat_a']



volume = volume_computation3(feat_t,feat_v,feat_a)

print("Volume: ", volume.detach().cpu())
```

## Download  VAST-27M annotations for pretraining

VAST-27M DATASET could be downloaded following the official [repo](https://github.com/TXH-mercury/VAST)

We used a subset of VAST-27M for the pretraining phase of GRAM. This is the annotation file used [here](https://drive.google.com/file/d/1s_YMQirx4MalnC_dw7h-NVY2KkTimCrC/view?usp=sharing)

## Finetune  Model on the 150k subset of VAST27M
Download annotations150k.json file subset.
Reference it in scripts/gram/finetune_ret.sh and in config/gram/finetune_cfg/pretrain-gram.json
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
--pretrain_dir $PATH-TO-CKPT-FOLDER \
--output_dir $PATH-WHERE-TO-STORE-RESULTS \
```

if you want to test model, just add following two rows to the cmd:
```
--mode 'testing' \
--checkpoint /PATH/TO/SAVED_CHECKPOINT.pt
```

## Citation

If you find this code useful for your research, please consider citing the following paper:

```
@inproceedings{cicchetti2025gramian,
title={Gramian Multimodal Representation Learning and Alignment},
author={Giordano Cicchetti and Eleonora Grassucci and Luigi Sigillo and Danilo Comminiello},
booktitle={The Thirteenth International Conference on Learning Representations},
year={2025},
url={https://openreview.net/forum?id=ftGnpZrW7P}
}
```

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=ispamm/GRAM&type=Date)](https://star-history.com/#ispamm/GRAM&Date)


## Third-Party Licenses

For the full list of third-party licenses used in this project, please see the [THIRD_PARTY_LICENSES.md](THIRD_PARTY_LICENSES.md) file.
