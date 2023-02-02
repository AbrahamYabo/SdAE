# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------

from functools import partial

import torch
import torch.nn as nn

import timm.models.vision_transformer


class VisionTransformer(timm.models.vision_transformer.VisionTransformer):
    """ Vision Transformer with support for global average pooling
    """
    def __init__(self, global_pool=False, mask_ratio=None, shrink_num=None, **kwargs):
        super(VisionTransformer, self).__init__(**kwargs)

        self.global_pool = global_pool
        del self.head  # remove the head of vision transformer
        if self.global_pool:
            norm_layer = kwargs['norm_layer']
            embed_dim = kwargs['embed_dim']
            self.fc_norm = norm_layer(embed_dim)

            del self.norm  # remove the original norm

    def forward_teachers(self, x, ids_unkeep=None, shrink_num=None, ncrop_loss=None):
        B = x.shape[0]
        x = self.patch_embed(x)

        if shrink_num is not None:
            ids_unkeep = ids_unkeep[:, : shrink_num]
        
        x = x + self.pos_embed[:, 1:, :]

        if shrink_num is not None and ids_unkeep is not None:
            if ncrop_loss is not None:
                if shrink_num % ncrop_loss == 0:
                    x = torch.gather(x, dim=1, index=ids_unkeep.unsqueeze(-1).repeat(1, 1, x.shape[2]))
                    x = x.reshape(B*ncrop_loss, int(x.shape[1]/ncrop_loss), x.shape[2])
                else:
                    shrink_num_tmp = shrink_num - shrink_num % ncrop_loss
                    ids_unkeep = ids_unkeep[:, : shrink_num_tmp]
                    x = torch.gather(x, dim=1, index=ids_unkeep.unsqueeze(-1).repeat(1, 1, x.shape[2]))
                    x = x.reshape(B*ncrop_loss, int(x.shape[1]/ncrop_loss), x.shape[2])
            else:
                #When ncrop_loss is None, the multi-fold strategy will not be operated. All masked tokens are fed without bundle.
                x = torch.gather(x, dim=1, index=ids_unkeep.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # keep only masked tokens

        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)
        return x

    def forward(self, x, ids_unkeep=None, shrink_num=None, ncrop_loss=None):
        x = self.forward_teachers(x, ids_unkeep, shrink_num, ncrop_loss)
        return x[:, 1:, :]

def vit_small_patch16(**kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True, 
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def vit_base_patch16(**kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def vit_large_patch16(**kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def vit_huge_patch14(**kwargs):
    model = VisionTransformer(
        patch_size=14, embed_dim=1280, depth=32, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model