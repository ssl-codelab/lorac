# LORAC: A Low Rank Promoting Prior for Unsupervised Contrastive Learning
This repository is the official [PyTorch](http://pytorch.org/) implementation of **LORAC** (A **LO**w **RA**nk Promoting Prior for Unsupervised **C**ontrastive Learning).

<div align="center">
  <img width="50%" alt="LORAC Framework" src="https://github.com/ssl-codelab/lorac/releases/download/v1.0.0/lorac-framework.png">
</div>



## 0 Requirements

- Python 3.6
- [PyTorch](http://pytorch.org) install = 1.6.0
- torchvision install = 0.7.0
- CUDA 10.1
- [Apex](https://github.com/NVIDIA/apex) with CUDA extension
- Other dependencies: opencv-python, scipy, pandas, numpy

## 1 Pretraining
We release a demo for several self-supervised learning approaches. These methods include: MoCo-M ([here](https://github.com/ssl-codelab/lorac/releases/download/v1.0.0/mocom_r50_e200_pretrained.pth.tar)) and LORAC ([here](https://github.com/ssl-codelab/lorac/releases/download/v1.0.0/lorac_r50_e200_pretrained.pth.tar)) pretrained models. All of the models are based on ResNet50 architecture, pretrained for 200 epochs.

### 1.1 MoCo-M pretrain

To train MoCo-M on a single node with 4 gpus for 200 epochs, run:

```shell
DATASET_PATH="path/to/ImageNet1K/train"
EXPERIMENT_PATH="path/to/experiment"

mkdir -p ${EXPERIMENT_PATH}
python -m torch.distributed.launch --nproc_per_node=4 main_moco_m.py \
--data_path ${DATASET_PATH} \
--nmb_crops 3 5 \
--size_crops 224 96 \
--min_scale_crops 0.14 0.05 \
--max_scale_crops 1. 0.14 \
--crops_for_assign 0 1 \
--moco-dim 128 \
--moco-k 65536 \
--moco-m 0.999 \
--temperature 0.2 \
--mlp true \
--epochs 200 \
--batch_size 64 \
--base_lr 1.8 \
--final_lr 0.018 \
--wd 1e-6 \
--warmup_epochs 0 \
--checkpoint_freq 1 \
--dist_url "tcp://localhost:40000" \
--arch mocom_r50 \
--use_fp16 true \
--sync_bn pytorch \
--dump_path ${EXPERIMENT_PATH}
```

### 1.2 LORAC pretrain

To train LORAC on a single node with 4 gpus for 200 epochs, run:

```shell
DATASET_PATH="path/to/ImageNet1K/train"
EXPERIMENT_PATH="path/to/experiment"

mkdir -p ${EXPERIMENT_PATH}
python -m torch.distributed.launch --nproc_per_node=4 main_lorac.py \
--data_path ${DATASET_PATH} \
--nmb_crops 3 5 \
--size_crops 224 96 \
--min_scale_crops 0.14 0.05 \
--max_scale_crops 1. 0.14 \
--crops_for_assign 0 1 \
--moco-dim 128 \
--moco-k 65536 \
--moco-m 0.999 \
--temperature 0.2 \
--mlp true \
--epochs 200 \
--batch_size 64 \
--base_lr 1.8 \
--final_lr 0.018 \
--lorac_beta 15. \
--epoch_lorac_starts 100 \
--wd 1e-6 \
--warmup_epochs 0 \
--checkpoint_freq 1 \
--dist_url "tcp://localhost:40000" \
--arch mocom_r50 \
--use_fp16 true \
--sync_bn pytorch \
--dump_path ${EXPERIMENT_PATH}
```

## 2 Linear Evaluation

To train a linear classifier on frozen features out of deep network pretrained via various self-supervised pretraining methods, run:
```shell
DATASET_PATH="path/to/ImageNet1K"
EXPERIMENT_PATH="path/to/experiment"
LINCLS_PATH="path/to/lincls"

python -m torch.distributed.launch --nproc_per_node=4 eval_linear.py \
--data_path ${DATASET_PATH} \
--arch resnet50 \
--lr 6.0 \
--scheduler_type step \
--dump_path ${LINCLS_PATH} \
--pretrained ${EXPERIMENT_PATH}/checkpoint.pth.tar \
--batch_size 512 \
--wd 0.0 \
--num_classes 1000
```

