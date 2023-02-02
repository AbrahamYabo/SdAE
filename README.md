# ECCV2022-SdAE
The implementation of "[SdAE: Self-distillated Masked Autoencoder](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136900107.pdf)", ECCV2022.


## Congifuration Environment
- Python 3.7
- Pytorch 1.7.0 
- torchvision 0.8.1
- timm 0.3.2
- PyYAML 


## Pre-training
The detailed pre-training instruction is in PRETRAIN.md.
Taking 300 epochs Vit-Base pretraining as example:
```
python -m torch.distributed.launch --nproc_per_node 8 main_pretrain.py \
        --batch_size 24 --epochs 300 \
        --model mae_vit_base_patch16 --model_teacher vit_base_patch16 --data_path {DATA_DIR} --warmup_epochs 60 --mask_ratio 0.75 \
        --blr 2.666e-4 --ema_op per_epoch --ema_frequent 1 \
        --momentum_teacher 0.96 --momentum_teacher_final 0.99 \
        --drop_path 0.25 --shrink_num 147 --ncrop_loss 3
```

## Fine-tuning
```
The fine-tuning instruction is in FINETUNE.md.
Taking Vit-Base fine-tuning as example:
python -m torch.distributed.launch --nproc_per_node 8 main_finetune.py --finetune {WEIGHT_DIR} \
        --batch_size 128 --epochs 100 --model  vit_base_patch16 --dist_eval --data_path {DATA_DIR} 
```

## Citation
Please cite our paper if the code is helpful to your research.
```
@inproceedings{liu2022source,
    author = {Yabo Chen, Yuchen Liu,  Dongsheng Jiang, Xiaopeng Zhang, Wenrui Dai, Hongkai Xiong and Qi Tian},
    title = {SdAE: Self-distillated masked autoencoder},
    booktitle = {ECCV},
    year = {2022}
}

