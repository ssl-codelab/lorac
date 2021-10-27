import argparse
import math
import os
import shutil
import time
from logging import getLogger

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import apex
from apex.parallel.LARC import LARC

from src.utils import (
    bool_flag,
    initialize_exp,
    restart_from_checkpoint,
    fix_random_seeds,
    AverageMeter,
    init_distributed_mode,
    PaceAverageMeter,
    accuracy,
)
from src.multicropdataset import MultiCropDataset
import models


logger = getLogger()

parser = argparse.ArgumentParser(description="Implementation of MoCo")

#########################
#### data parameters ####
#########################
parser.add_argument("--data_path", type=str, default="/path/to/imagenet",
                    help="path to dataset repository")
parser.add_argument("--nmb_crops", type=int, default=[2], nargs="+",
                    help="list of number of crops (example: [2, 6])")
parser.add_argument("--size_crops", type=int, default=[224], nargs="+",
                    help="crops resolutions (example: [224, 96])")
parser.add_argument("--min_scale_crops", type=float, default=[0.14], nargs="+",
                    help="argument in RandomResizedCrop (example: [0.14, 0.05])")
parser.add_argument("--max_scale_crops", type=float, default=[1], nargs="+",
                    help="argument in RandomResizedCrop (example: [1., 0.14])")
parser.add_argument("--crops_for_assign", type=int, nargs="+", default=[0, 1],
                    help="list of crops id used for computing assignments")
parser.add_argument("--color_distortion_scale", type=float, default=1.0,
                    help="argument in RandomResizedCrop (example: [1., 0.14])")

#########################
## moco specific params #
#########################
parser.add_argument('--moco-dim', default=128, type=int,
                    help='feature dimension (default: 128)')
parser.add_argument('--moco-k', default=65536, type=int,
                    help='queue size; number of negative keys (default: 65536)')
parser.add_argument('--moco-m', default=0.999, type=float,
                    help='moco momentum of updating key encoder (default: 0.999)')
parser.add_argument('--mlp', type=bool_flag, default=True,
                    help='use mlp head')
parser.add_argument('--temperature', default=0.2, type=float,
                    help='softmax temperature (default: 0.2)')

##############################
### lorac specific configs ###
##############################
parser.add_argument("--lorac_beta", type=float, default=15.,
                    help="lorac beta (example: 15)")
parser.add_argument("--epoch_lorac_starts", type=int, default=100,
                    help="from this epoch, we start using the lowrank prior")

#########################
#### optim parameters ###
#########################
parser.add_argument("--epochs", default=100, type=int,
                    help="number of total epochs to run")
parser.add_argument("--batch_size", default=64, type=int,
                    help="batch size per gpu, i.e. how many unique instances per gpu")
parser.add_argument("--base_lr", default=4.8, type=float, help="base learning rate")
parser.add_argument("--final_lr", type=float, default=0, help="final learning rate")
parser.add_argument("--wd", default=1e-6, type=float, help="weight decay")
parser.add_argument("--warmup_epochs", default=10, type=int, help="number of warmup epochs")
parser.add_argument("--start_warmup", default=0, type=float,
                    help="initial warmup learning rate")

#########################
#### dist parameters ###
#########################
parser.add_argument("--dist_url", default="env://", type=str, help="""url used to set up distributed
                    training; see https://pytorch.org/docs/stable/distributed.html""")
parser.add_argument("--world_size", default=-1, type=int, help="""
                    number of processes: it is set automatically and
                    should not be passed as argument""")
parser.add_argument("--rank", default=0, type=int, help="""rank of this process:
                    it is set automatically and should not be passed as argument""")
parser.add_argument("--local_rank", default=0, type=int,
                    help="this argument is not used and should be ignored")

#########################
#### other parameters ###
#########################
parser.add_argument("--arch", default="resnet50", type=str, help="convnet architecture")
parser.add_argument("--hidden_mlp", default=2048, type=int,
                    help="hidden layer dimension in projection head")
parser.add_argument("--workers", default=10, type=int,
                    help="number of data loading workers")
parser.add_argument("--checkpoint_freq", type=int, default=10,
                    help="Save the model periodically")
parser.add_argument("--use_fp16", type=bool_flag, default=True,
                    help="whether to train with mixed precision or not")
parser.add_argument("--sync_bn", type=str, default="pytorch", help="synchronize bn")
parser.add_argument("--syncbn_process_group_size", type=int, default=8, help=""" see
                    https://github.com/NVIDIA/apex/blob/master/apex/parallel/__init__.py#L58-L67""")
parser.add_argument("--dump_path", type=str, default=".",
                    help="experiment dump path for checkpoints and log")
parser.add_argument("--seed", type=int, default=31, help="seed")
parser.add_argument('--resume', default=None, type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')


def main():
    global args
    args = parser.parse_args()
    init_distributed_mode(args)
    fix_random_seeds(args.seed)
    logger, training_stats = initialize_exp(args, "epoch", "loss")

    # build data
    train_dataset = MultiCropDataset(
        args.data_path,
        args.size_crops,
        args.nmb_crops,
        args.min_scale_crops,
        args.max_scale_crops,
        color_distortion_scale=args.color_distortion_scale,
    )
    sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        sampler=sampler,
        batch_size=args.batch_size,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=True
    )
    if args.rank == 0:
        logger.info("Building data done with {} images loaded.".format(len(train_dataset)))

    # build model
    model = models.__dict__[args.arch](
        dim=args.moco_dim,
        K=args.moco_k,
        m=args.moco_m,
        mlp=args.mlp,
    )
    # synchronize batch norm layers
    if args.sync_bn == "pytorch":
        model.encoder_q = nn.SyncBatchNorm.convert_sync_batchnorm(model.encoder_q)
    elif args.sync_bn == "apex":
        # with apex syncbn we sync bn per group because it speeds up computation
        # compared to global syncbn
        process_group = apex.parallel.create_syncbn_process_group(args.syncbn_process_group_size)
        model.encoder_q = apex.parallel.convert_syncbn_model(model.encoder_q, process_group=process_group)
    # copy model to GPU
    model = model.cuda()
    if args.rank == 0:
        logger.info(model)
        logger.info("Building model done.")

    # build optimizer
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=args.base_lr,
        momentum=0.9,
        weight_decay=args.wd,
    )
    optimizer = LARC(optimizer=optimizer, trust_coefficient=0.001, clip=False)
    warmup_lr_schedule = np.linspace(args.start_warmup, args.base_lr, len(train_loader) * args.warmup_epochs)
    iters = np.arange(len(train_loader) * (args.epochs - args.warmup_epochs))
    cosine_lr_schedule = np.array([args.final_lr + 0.5 * (args.base_lr - args.final_lr) * (1 + \
                         math.cos(math.pi * t / (len(train_loader) * (args.epochs - args.warmup_epochs)))) for t in iters])
    lr_schedule = np.concatenate((warmup_lr_schedule, cosine_lr_schedule))
    if args.rank == 0:
        logger.info("Building optimizer done.")

    # init mixed precision
    if args.use_fp16:
        model, optimizer = apex.amp.initialize(model, optimizer, opt_level="O1")
        if args.rank == 0:
            logger.info("Initializing mixed precision done.")

    # wrap model
    model = nn.parallel.DistributedDataParallel(
        model,
        device_ids=[args.gpu_to_work_on]
    )

    # optionally resume from a checkpoint
    to_restore = {"epoch": 0}
    resume_path = os.path.join(args.dump_path, "checkpoint.pth.tar")
    if args.resume is not None:
        resume_path = args.resume
    restart_from_checkpoint(
        resume_path,
        run_variables=to_restore,
        state_dict=model,
        optimizer=optimizer,
        amp=apex.amp,
    )
    start_epoch = to_restore["epoch"]

    cudnn.benchmark = True

    for epoch in range(start_epoch, args.epochs):

        # train the network for one epoch
        if args.rank == 0:
            logger.info("============ Starting epoch %i ... ============" % epoch)

        # set sampler
        train_loader.sampler.set_epoch(epoch)

        # train the network
        scores = train(train_loader, model, optimizer, epoch, lr_schedule)
        training_stats.update(scores)

        # save checkpoints
        if args.rank == 0:
            save_dict = {
                "epoch": epoch + 1,
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
            }
            if args.use_fp16:
                save_dict["amp"] = apex.amp.state_dict()
            torch.save(
                save_dict,
                os.path.join(args.dump_path, "checkpoint.pth.tar"),
            )
            if (epoch + 1) % args.checkpoint_freq == 0 or (epoch + 1) == args.epochs:
                shutil.copyfile(
                    os.path.join(args.dump_path, "checkpoint.pth.tar"),
                    os.path.join(args.dump_checkpoints, "ckp-" + str(epoch) + ".pth"),
                )


def train(train_loader, model, optimizer, epoch, lr_schedule):
    left_time = AverageMeter()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = PaceAverageMeter()
    ranks = PaceAverageMeter()
    top1 = PaceAverageMeter(pace=200)
    top5 = PaceAverageMeter(pace=200)

    model.train()

    end = time.time()
    sample_number = len(train_loader)

    for it, inputs in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        # update learning rate
        iteration = epoch * len(train_loader) + it
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr_schedule[iteration]

        # ============ multi-res forward passes ... ============
        q, k, queue = model(inputs)
        bs = inputs[0].size(0)
        total_crop = np.sum(args.nmb_crops)

        # ============ lorac loss ... ============
        loss = 0
        k_list = k.split([bs for _ in range(len(args.crops_for_assign))], dim=0)
        for i, crop_id in enumerate(args.crops_for_assign):
            # lorac loss with multi-crop augmentation
            _q = q.reshape(total_crop, bs, -1)[np.delete(np.arange(total_crop), crop_id), ...].reshape(-1, q.shape[1])
            _k = k_list[crop_id].repeat(total_crop - 1, 1)
            l_pos = torch.einsum('nc,nc->n', [_q, _k]).unsqueeze(-1)
            l_neg = torch.einsum('nc,ck->nk', [_q, queue[1 - crop_id]])
            logits = torch.cat([l_pos, l_neg], dim=1) / args.temperature
            if epoch < args.epoch_lorac_starts:
                rank = q.sum() * 0.
                logits_lowrank = logits
            else:
                # compute rank
                feat_dim = q.size(1)
                qk_mat = torch.cat([
                    q.reshape(-1, bs, feat_dim)[:args.nmb_crops[0], ...],
                    k.reshape(-1, bs, feat_dim)], dim=0)
                rank = torch.norm(qk_mat, p='nuc', dim=(0, 2)).unsqueeze(dim=1)
                l_pos_lowrank = l_pos - (rank.repeat(l_pos.shape[0] // bs, 1) / args.lorac_beta)
                logits_lowrank = torch.cat([l_pos_lowrank, l_neg], dim=1)
                rank = rank.mean()
            labels = torch.zeros(logits_lowrank.shape[0], dtype=torch.long).cuda()
            loss += F.cross_entropy(logits_lowrank, labels) / len(args.crops_for_assign)

        # ============ training acc ... ============
        acc1, acc5 = accuracy(logits, labels, topk=(1, 5))
        top1.update(acc1[0])
        top5.update(acc5[0])

        # ============ backward and optim step ... ============
        optimizer.zero_grad()
        if args.use_fp16:
            with apex.amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()
        optimizer.step()

        # ============ misc ... ============
        losses.update(loss.item())
        ranks.update(rank.item())
        batch_time.update(time.time() - end)
        end = time.time()
        if args.rank == 0 and (it + 1) % 100 == 0:
            logger.info(
                "Epoch: [{0}][{1}/{2}]\t"
                "Loss {loss.val:.4f} ({loss.avg:.4f})\t"
                "Rank {rank.val:.4f} ({rank.avg:.4f})\t"
                "Acc@1 {top1.val:.4f} ({top1.avg:.4f})\t"
                "Acc@5 {top5.val:.4f} ({top5.avg:.4f})\t"
                "Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t"
                "Data {data_time.val:.3f} ({data_time.avg:.3f})\t"
                "Lr: {lr:.4f}".format(
                    epoch,
                    it + 1,
                    sample_number,
                    loss=losses,
                    rank=ranks,
                    top1=top1,
                    top5=top5,
                    batch_time=batch_time,
                    data_time=data_time,
                    lr=optimizer.optim.param_groups[0]["lr"],
                )
            )
            iter = epoch * sample_number + it
    return (epoch, losses.avg)


if __name__ == "__main__":
    main()
