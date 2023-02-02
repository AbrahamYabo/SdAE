# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------
import math
import sys
from typing import Iterable

import torch

import util.misc as misc
import util.lr_sched as lr_sched
import torch.nn.functional as F
import copy
from util.pos_embed import interpolate_pos_embed

from collections import OrderedDict

def patchify(model, imgs):
    """
    imgs: (N, 3, H, W)
    x: (N, L, patch_size**2 *3)
    """
    p = model.module.patch_embed.patch_size[0]
    assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

    h = w = imgs.shape[2] // p
    x = imgs.reshape(shape=(imgs.shape[0], 3, h, p, w, p))
    x = torch.einsum('nchpwq->nhwpqc', x)
    x = x.reshape(shape=(imgs.shape[0], h * w, p**2 * 3))
    return x

def weight_delivery(model, epoch, momentum_schedule):
    with torch.no_grad():
        m = momentum_schedule[epoch]  # momentum parameter
        student = model
        all_keys = list(student.module.state_dict().keys())
        new_dict = OrderedDict()
        for key in all_keys:
            if key.startswith('backbone.'):
                new_dict[key[9:]] = student.module.state_dict()[key]
            elif key.startswith('encoder.') and not 'norm.' in key:
                new_dict[key[8:]] = student.module.state_dict()[key]
            else:
                if key.startswith('decoder') or key.startswith('encoder_to_decoder.') or key.startswith('mask_token') or key.startswith('encoder.norm'):
                    pass
                else:
                    new_dict[key] = student.module.state_dict()[key]
    return new_dict


def EMA_process(model, model_teacher, model_teacher_without_ddp, epoch, momentum_schedule):
    # EMA update for the teacher
    student_crop = copy.deepcopy(model_teacher_without_ddp)
    m = momentum_schedule[epoch]
    with torch.no_grad():
        new_dict = weight_delivery(model, epoch, momentum_schedule)
        #interpolate_pos_embed(student_crop, new_dict)
        student_crop.load_state_dict(new_dict)
        for param_q, param_k in zip(student_crop.parameters(), model_teacher_without_ddp.parameters()):
            param_k.data.mul_(m).add_((1 - m) * param_q.detach().data)

def train_one_epoch_ema(model: torch.nn.Module, model_teacher: torch.nn.Module, model_teacher_without_ddp: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler,
                    log_writer=None, momentum_schedule=None, ema_op=None, is_ema=None, shrink_num=None, ncrop_loss=None,
                    args=None):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 50

    accum_iter = args.accum_iter

    optimizer.zero_grad()

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))

    for data_iter_step, (samples, _) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):

        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        samples = samples.to(device, non_blocking=True)

        with torch.cuda.amp.autocast():
            outputs, mask, ids_unkeep = model(samples, mask_ratio=args.mask_ratio)

            with torch.no_grad():
                output_teacher = model_teacher(samples, ids_unkeep=ids_unkeep, shrink_num=shrink_num, ncrop_loss=ncrop_loss)

            mean = output_teacher.mean(dim=-1, keepdim=True)
            var = output_teacher.var(dim=-1, keepdim=True)
            output_teacher = (output_teacher - mean) / (var + 1.e-6)**.5
            
            if shrink_num is not None: 
                if ncrop_loss is not None:
                    if shrink_num % ncrop_loss == 0:
                        outputs = torch.gather(outputs, dim=1, index=ids_unkeep.unsqueeze(-1).repeat(1, 1, outputs.shape[2]))
                        outputs = outputs.reshape(outputs.shape[0]*ncrop_loss, int(outputs.shape[1]/ncrop_loss), outputs.shape[2])
                    else:
                        shrink_num_tmp = shrink_num - shrink_num % ncrop_loss
                        ids_unkeep = ids_unkeep[:, : shrink_num_tmp]
                        outputs = torch.gather(outputs, dim=1, index=ids_unkeep.unsqueeze(-1).repeat(1, 1, outputs.shape[2]))
                        outputs = outputs.reshape(outputs.shape[0]*ncrop_loss, int(outputs.shape[1]/ncrop_loss), outputs.shape[2])
                else:
                    #When ncrop_loss is None, the multi-fold strategy will not be operated. All masked tokens are fed without bundle.
                    ids_unkeep = ids_unkeep[:, : shrink_num]
                    outputs = torch.gather(outputs, dim=1, index=ids_unkeep.unsqueeze(-1).repeat(1, 1, outputs.shape[2]))  # keep only masked tokens
            else:
                #When shrink_num is None, all the tokens(196 if set patch size 16 for 224x224 images) will be sent to the teacher branch
                bool_masked_pos = mask.to(outputs.device, non_blocking=True).flatten(1).to(torch.bool)
                outputs = outputs[bool_masked_pos]
                output_teacher = output_teacher[bool_masked_pos]
            
            loss = 1 - F.cosine_similarity(output_teacher, outputs).mean()
            loss_value = loss.item()

            if not math.isfinite(loss_value):
                print("Loss is {}, stopping training".format(loss_value))
                sys.exit(1)

            loss /= accum_iter
            loss_scaler(loss, optimizer, parameters=model.parameters(),
                        update_grad=(data_iter_step + 1) % accum_iter == 0)
            if (data_iter_step + 1) % accum_iter == 0:
                optimizer.zero_grad()

        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)

        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=lr)

        loss_value_reduce = misc.all_reduce_mean(loss_value)
        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            """ We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar('train_loss', loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('lr', lr, epoch_1000x)

        
        if 'per_ite' in ema_op:
            #Process EMA for the teacher every iteration. The momentum needs to be much smaller(more closer to 1)
            EMA_process(model, model_teacher, model_teacher_without_ddp, epoch, momentum_schedule)
    
    if not 'per_ite' in ema_op:
        if is_ema is True:
            #Process EMA for the teacher every epoch.
            EMA_process(model, model_teacher, model_teacher_without_ddp, epoch, momentum_schedule)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

