# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import torch
import torch.nn as nn
from .backbones import resnet18, resnet50
from src.utils import concat_all_gather

class MoCoM(nn.Module):
    """
    Build a MoCo model with: a query encoder, a key encoder, and a queue
    https://arxiv.org/abs/1911.05722
    """
    def __init__(self, base_encoder, dim=128, K=65536, m=0.999, mlp=False):
        """
        dim: feature dimension (default: 128)
        K: queue size; number of negative keys (default: 65536)
        m: moco momentum of updating key encoder (default: 0.999)
        T: softmax temperature (default: 0.07)
        """
        super(MoCoM, self).__init__()

        self.K = K
        self.m = m

        # create the encoders
        # num_classes is the output fc dimension
        self.encoder_q = base_encoder(num_classes=dim)
        self.encoder_k = base_encoder(num_classes=dim)

        if mlp:  # hack: brute-force replacement
            dim_mlp = self.encoder_q.fc.weight.shape[1]
            self.encoder_q.fc = nn.Sequential(
                nn.Linear(dim_mlp, dim_mlp),
                nn.ReLU(),
                self.encoder_q.fc)
            self.encoder_k.fc = nn.Sequential(
                nn.Linear(dim_mlp, dim_mlp),
                nn.ReLU(),
                self.encoder_k.fc)

        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

        # create the queue
        self.register_buffer("queue", torch.randn(2, dim, K))
        self.queue = nn.functional.normalize(self.queue, dim=1)

        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        # gather keys before updating queue
        keys = concat_all_gather(keys)
        num_keys_per_gpu = keys.size(0) // (torch.distributed.get_world_size() * 2)
        keys_list = keys.split(
            [num_keys_per_gpu for _ in range(torch.distributed.get_world_size() * 2)], dim=0)

        batch_size = keys.shape[0] // 2

        ptr = int(self.queue_ptr)
        assert self.K % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.queue[0, :, ptr:ptr + batch_size] = torch.cat(keys_list[0::2], dim=0).T
        self.queue[1, :, ptr:ptr + batch_size] = torch.cat(keys_list[1::2], dim=0).T
        ptr = (ptr + batch_size) % self.K  # move pointer

        self.queue_ptr[0] = ptr

    @torch.no_grad()
    def _batch_shuffle_ddp(self, x):
        """
        Batch shuffle, for making use of BatchNorm.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # random shuffle index
        idx_shuffle = torch.randperm(batch_size_all).cuda()

        # broadcast to all gpus
        torch.distributed.broadcast(idx_shuffle, src=0)

        # index for restoring
        idx_unshuffle = torch.argsort(idx_shuffle)

        # shuffled index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_shuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this], idx_unshuffle

    @torch.no_grad()
    def _batch_unshuffle_ddp(self, x, idx_unshuffle):
        """
        Undo batch shuffle.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # restored index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_unshuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this]

    def forward_q(self, x):
        return self.encoder_q(x)

    def forward_k(self, x):
        with torch.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder()  # update the key encoder
            # shuffle for making use of BN
            im_k, idx_unshuffle = self._batch_shuffle_ddp(x)
            k = self.encoder_k(im_k)  # keys: NxC
            # undo shuffle
            k = self._batch_unshuffle_ddp(k, idx_unshuffle)
        return k

    def forward_head(self, q, k):
        q = nn.functional.normalize(q, dim=1, p=2)
        k = nn.functional.normalize(k, dim=1, p=2)
        queue = self.queue.clone().detach()
        # dequeue and enqueue
        self._dequeue_and_enqueue(k)
        return q, k, queue

    def forward(self, inputs, eval_mode=False):
        if not isinstance(inputs, list):
            inputs = [inputs]
        idx_crops = torch.cumsum(torch.unique_consecutive(
            torch.tensor([inp.shape[-1] for inp in inputs]),
            return_counts=True,
        )[1], 0)
        bs = inputs[0].shape[0]
        start_idx = 0
        for end_idx in idx_crops:
            img = torch.cat(inputs[start_idx: end_idx]).cuda(non_blocking=True)
            _q = self.forward_q(img)
            if start_idx == 0:
                q = _q
                k = self.forward_k(img[start_idx: start_idx+bs*2, ...])
            else:
                q = torch.cat((q, _q))
            start_idx = end_idx
        if eval_mode:
            return q
        return self.forward_head(q, k)


def mocom_r18(**kwargs):
    return MoCoM(resnet18, **kwargs)


def mocom_r50(**kwargs):
    return MoCoM(resnet50, **kwargs)