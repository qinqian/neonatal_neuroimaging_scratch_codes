#!/bin/bash -ex

#python train_unet.py --seed 99 -gpu 1 --shape 64 -batch 16 --model_dir focal_loss_seed9_shape64_batch_16 --loss bce --schedule 1

#python train_unet.py --seed 99 -gpu 2 --shape 64 -batch 32 --model_dir focal_loss_seed9_shape64_batch_16 --loss bce --opt ams --schedule 1

#python train_unet.py --seed 99 -gpu 1 --shape 64 -batch 32 --model_dir focal_loss_seed9_shape64_batch_16 --loss focal --opt ams --schedule 1

#python train_unet.py --seed 99 -gpu 0 --shape 64 -batch 32 --model_dir focal_loss_seed9_shape64_batch_16 --loss bce --opt ams 

## white matter hyper
#python train_unet.py --seed 99 -gpu 1 --lr 0.000005 --shape 64 -batch 64 --model_dir focal_loss_seed9_shape64_batch_16 --loss bce --opt ams  --model edd

#python train_unet.py --seed 99 -gpu 2 --lr 0.0005 --shape 64 -batch 64 --model_dir focal_loss_seed9_shape64_batch_16 --loss bce --opt ams  --model edd --schedule 1

#python train_unet.py --seed 99 -gpu 0 --lr 0.001 --shape 128 -batch 64 --model_dir focal_loss_seed9_shape64_batch_16 --loss bce --opt ams  --model edd --schedule 1

#python train_unet.py --seed 99 -gpu 0 --lr 0.0005 --decay 0.1 --shape 128 -batch 16 --model_dir focal_loss_seed9_shape64_batch_16 --loss bce --opt sgd --model edd --modality 1 --schedule decay

#python train_unet.py --seed 99 -gpu 1 --lr 0.00001 --shape 128 -batch 16 --model_dir focal_loss_seed9_shape64_batch_16 --loss dice_loss --opt ams  --model edd --schedule plat --modality 3

#python train_unet.py --seed 99 -gpu 0 --lr 0.00002 --shape 32 -batch 16 --model_dir focal_loss_seed9_shape64_batch_16 --loss dice_loss --opt ams --model edd --schedule decay --modality 3

#python train_unet.py --seed 99 -gpu 0 --lr 0.0001 --shape 32 -batch 16 --model_dir focal_loss_seed9_shape64_batch_16 --loss bce --opt ams --model deeplab --schedule decay --modality 3

#python train_unet.py --seed 99 -gpu 0 --lr 0.00002 --shape 32 -batch 16 --model_dir focal_loss_seed9_shape64_batch_16 --loss bce --opt ams --model edd --schedule decay --modality 3

python train_unet.py --seed 99 -gpu 0 --lr 0.0001 --shape 32 -batch 16 --model_dir focal_loss_seed9_shape64_batch_16 --loss bce --opt ams --model deeplab --schedule decay --modality 3

