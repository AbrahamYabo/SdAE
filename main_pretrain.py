# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------
import argparse
import datetime
import json
import numpy as np
import os
import time
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms
import torchvision.datasets as datasets

import timm

assert timm.__version__ == "0.3.2"  # version check
import timm.optim.optim_factory as optim_factory

import util.misc as misc
from util.misc import NativeScalerWithGradNormCount as NativeScaler
from util.pos_embed import interpolate_pos_embed
from util.cosine_scheduler import cosine_scheduler

import models_mae
import models_teacher


from engine_pretrain import train_one_epoch_ema


def get_args_parser():
    parser = argparse.ArgumentParser('MAE pre-training', add_help=False)
    parser.add_argument('--batch_size', default=96, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus, Smaller batch size(e.g. 96 batch x 8 gpu) can produce better performance')
    parser.add_argument('--epochs', default=300, type=int)
    parser.add_argument('--accum_iter', default=1, type=int,
                        help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')

    # Model parameters
    parser.add_argument('--model', default='mae_vit_base_patch16', type=str, metavar='MODEL',
                        help='Name of model to train')

    parser.add_argument('--input_size', default=224, type=int,
                        help='images input size')
    
    #parser.add_argument('--tea_input_size', default=112, type=int,
    #                    help='images input size')

    parser.add_argument('--mask_ratio', default=0.75, type=float,
                        help='Masking ratio (percentage of removed patches).')

    parser.add_argument('--norm_pix_loss', action='store_true',
                        help='Use (per-patch) normalized pixels as targets for computing loss')
    parser.set_defaults(norm_pix_loss=True)

    # Optimizer parameters
    parser.add_argument('--weight_decay', type=float, default=0.04, help="""Initial value of the
        weight decay. With ViT, a smaller value at the beginning of training works well.""")
    parser.add_argument('--weight_decay_end', type=float, default=0.4, help="""Final value of the
        weight decay. We use a cosine schedule for WD and using a larger decay by
        the end of training improves performance for ViTs.""")

    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='learning rate (absolute lr)')
    parser.add_argument('--blr', type=float, default=2.666e-4, metavar='LR',
                        help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--min_lr', type=float, default=0., metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')

    parser.add_argument('--warmup_epochs', type=int, default=60, metavar='N',
                        help='epochs to warmup LR, We set warmup epochs as 0.2 * epochs')

    # Models to produce features for decoder
    parser.add_argument('--model_teacher_path', default=None, help='the path of teachers checkpoint')
    parser.add_argument('--model_key', default='model|module', type=str)
    parser.add_argument('--model_prefix', default='', type=str)
    parser.add_argument('--model_teacher', default='vit_base_patch16', type=str, metavar='MODEL',
                        help='Name of model to train')
    parser.add_argument('--nb_classes', default=1000, type=int,
                        help='number of the classification types')
    parser.add_argument('--drop', type=float, default=0.0, metavar='PCT',
                        help='Dropout rate (default: 0.)')
    parser.add_argument('--drop_path', type=float, default=0.25, metavar='PCT',
                        help='Drop path rate (default: 0.25)')
    parser.add_argument('--attn_drop_rate', type=float, default=0.0, metavar='PCT',
                        help='Attention dropout rate (default: 0.)')
    parser.add_argument('--use_mean_pooling', action='store_true')
    parser.set_defaults(use_mean_pooling=True)
    parser.add_argument('--init_scale', default=0.001, type=float)
    parser.add_argument('--global_pool', action='store_true')
    parser.set_defaults(global_pool=False)
    parser.add_argument('--momentum_teacher', default=0.96, type=float, help="""Base EMA
        parameter for teacher update. The value is increased to [momentum_teacher_final] during training with cosine schedule.
        We recommend setting a higher value with small batches: for example use 0.96 to 0.99 with batch size of 256.""")
    parser.add_argument('--momentum_teacher_final', default=0.99, type=float, help="""The end value of base EMA
        parameter for teacher update. We recommend setting a higher value with small batches: for example use 0.96 with batch size of 256.""")
    parser.add_argument('--momentum_teacher_warmup', default=0.0, type=float, help="""Only worked when momentum_teacher_warmup_ep > 0, The EMA
        parameter for teacher update. The value is increased from [momentum_teacher_warmup] to [momentum_teacher] in the first [momentum_teacher_warmup_ep] epochs""")
    parser.add_argument('--momentum_teacher_warmup_ep', default=0, type=int, help="""Number of momentum warmup epochs""")

    parser.add_argument('--ema_op', default='per_epoch', type=str)
    parser.add_argument('--ema_frequent', default=1, type=int, help='how frequent the ema do')

    parser.add_argument('--shrink_num', default=None, type=int, help="""number of tokens feed into the model teacher. 
        Shrink_num indicate how many tokens are sent to the teachers. Pay attention, there should be 0 < shrink_num <= num_of_tokens * mask_ratio
        When shrink_num is None, all the tokens(196 if set patch size 16 for 224x224 images) will be sent to the teacher branch""")
    parser.add_argument('--ncrop_loss', default=None, type=int, help="""number of multi fold strategy in the model teacher
        When ncrop_loss is None, the multi-fold strategy will not be operated. All masked tokens are fed without bundle.""")

    # Dataset parameters
    parser.add_argument('--data_path', default='/cache/imagenet/', type=str, help='The path of imagenet. Make sure there exists [data_path]/train and [data_path]/val') 

    parser.add_argument('--output_dir', default='/cache/output/',
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default='/cache/output/',                      
                        help='path where to tensorboard log')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='',
                        help='resume from checkpoint')

    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')

    return parser


def main(args):
    misc.init_distributed_mode(args)

    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True

    # simple augmentation
    transform_train = transforms.Compose([
            transforms.RandomResizedCrop(args.input_size, scale=(0.2, 1.0), interpolation=3),  # 3 is bicubic
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    dataset_train = datasets.ImageFolder(os.path.join(args.data_path, 'train'), transform=transform_train)
    print(dataset_train)

    if True:  # args.distributed:
        num_tasks = misc.get_world_size()
        global_rank = misc.get_rank()
        sampler_train = torch.utils.data.DistributedSampler(
            dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
        )
        print("Sampler_train = %s" % str(sampler_train))
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)

    if global_rank == 0 and args.log_dir is not None:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = SummaryWriter(log_dir=args.log_dir)
    else:
        log_writer = None

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )
    
    #momentum_schedule = cosine_scheduler(args.momentum_teacher, 1,
    #                                           args.epochs, len(data_loader_train))
    #momentum_schedule = cosine_scheduler(args.momentum_teacher, args.momentum_teacher_final,
    #                                           args.epochs, 1)
    momentum_schedule = cosine_scheduler(base_value=args.momentum_teacher, 
                                        final_value=args.momentum_teacher_final, 
                                        epochs=args.epochs, 
                                        niter_per_ep=1, 
                                        warmup_epochs=args.momentum_teacher_warmup_ep, 
                                        start_warmup_value=args.momentum_teacher_warmup)
    
    print("momentum_schedule " + str(momentum_schedule))

    # define the model
    model = models_mae.__dict__[args.model](norm_pix_loss=args.norm_pix_loss, drop_path=args.drop_path)
    model_teacher = models_teacher.__dict__[args.model_teacher](
        num_classes=args.nb_classes,
        drop_path_rate=0.0,
        mask_ratio=args.mask_ratio,
        shrink_num=args.shrink_num,
    )

    model.to(device)

    if args.model_teacher_path:
        checkpoint = torch.load(args.model_teacher_path, map_location='cpu')

        print("Load pre-trained checkpoint from: %s" % args.model_teacher_path)
        checkpoint_model = checkpoint['model']
        state_dict = model_teacher.state_dict()
        for k in ['head.weight', 'head.bias']:
            if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
                print(f"Removing key {k} from pretrained checkpoint")
                del checkpoint_model[k]

        # interpolate position embedding
        interpolate_pos_embed(model_teacher, checkpoint_model)

        # load pre-trained model
        msg = model_teacher.load_state_dict(checkpoint_model, strict=False)
        print('msg:', msg)

    model_teacher.to(device)

    model_without_ddp = model
    model_teacher_without_ddp = model
    print("Model = %s" % str(model_without_ddp))

    eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()
    print('get_world_size:', misc.get_world_size())
    
    if args.lr is None:  # only base_lr is specified
        args.lr = args.blr * eff_batch_size / 256

    print("base lr: %.2e" % (args.lr * 256 / eff_batch_size))
    print("actual lr: %.2e" % args.lr)

    print("accumulate grad iterations: %d" % args.accum_iter)
    print("effective batch size: %d" % eff_batch_size)

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        model_teacher = torch.nn.parallel.DistributedDataParallel(model_teacher, device_ids=[args.gpu], find_unused_parameters=True)
        model_without_ddp = model.module
        model_teacher_without_ddp = model_teacher.module
    
    # following timm: set wd as 0 for bias and norm layers
    param_groups = optim_factory.add_weight_decay(model_without_ddp, args.weight_decay)
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95))
    print(optimizer)
    loss_scaler = NativeScaler()

    misc.load_model(args=args, model_without_ddp=model_without_ddp, 
                                optimizer=optimizer, 
                                loss_scaler=loss_scaler,
                                model_teacher_without_ddp=model_teacher_without_ddp)

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)

        if 'per_ite' in args.ema_op: args.ema_frequent = 1
        train_stats = train_one_epoch_ema(
            model, model_teacher, model_teacher_without_ddp, data_loader_train,
            optimizer, device, epoch, loss_scaler,
            log_writer=log_writer, momentum_schedule=momentum_schedule, 
            ema_op=args.ema_op, is_ema=True if (epoch+1) % args.ema_frequent == 0 else False, 
            shrink_num=args.shrink_num, ncrop_loss=args.ncrop_loss,
            args=args
        )
        if args.output_dir and (epoch % 20 == 0 or epoch + 1 == args.epochs):
            #Save the checkpoint every 20 epochs
            misc.save_model(
                args=args, model=model, model_without_ddp=model_without_ddp, model_teacher=model_teacher,
                model_teacher_without_ddp=model_teacher_without_ddp, optimizer=optimizer,
                loss_scaler=loss_scaler, epoch=epoch)

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                        'epoch': epoch,}

        if args.output_dir and misc.is_main_process():
            if log_writer is not None:
                log_writer.flush()
            with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
