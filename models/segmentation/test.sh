
#python train_unet.py --seed 99 -gpu 2 --shape 64 --lr 2e-4 -batch 16 --model_dir deeplab3plusfocal_loss_seed9_shape64_batch_16 --loss double --opt rms --decay 0.1 --model dense --schedule 1

python train_unet.py --seed 99 -gpu 0 --shape 64 --lr 1e-3 -batch 32 --model_dir eddplusfocal_loss_seed9_shape64_batch_16 --loss bce --opt ams --decay 0.1 --model edd --schedule 1
