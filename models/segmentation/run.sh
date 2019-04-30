#!/bin/bash 
plot_contour() {
    cp /data1/qinqian/MRI/01_PVL_AI/phaseA_exploration/remodel_mxnet/mask/dice_coef.xls .
    
    python mask_contour.py --t2flair ../cleaned_data/pvl/flair/pvl_142_PA342/sub-142_run-01_T2w_corrected.nii.gz --mask ../phaseA_exploration/remodel_mxnet/mask/all/chen_labels/1-3/142.nii.gz --mask2 ../phaseA_exploration/remodel_mxnet/mask/all/qq_labels/1-3/142.nii.gz --name 142_flair
    
    python mask_contour.py --t2flair ../cleaned_data/pvl/align/pvl_142_PA342/_intra_align0/sub-142_run-01_T1w_corrected_flirt.nii.gz --mask ../phaseA_exploration/remodel_mxnet/mask/all/chen_labels/1-3/142.nii.gz --mask2 ../phaseA_exploration/remodel_mxnet/mask/all/qq_labels/1-3/142.nii.gz --name 142_t1
}

train() {
   lr=0.00002
   model=edd
   seed=999 ## this division is good
   i=999 ## this division is good
   modality=3
   #for k in bce double focal; do
   #for k in focal bce; do

   SRC=`pwd`
   for k in bce; do
       gpu=1
       #for b in 16 8; do
       #for seed in 9 99 999; do
            for j in 64 96 32; do
            #for j in 96; do
                if [ $j -gt 64 ];then
                    #b=8 ## all change to 16
                    b=16 ## all change to 16
                else
                    b=16
                fi
                echo $j
                #(
                #  python train_unet.py --seed $seed -gpu $gpu --lr $lr --shape $j -batch $b --model_dir lr_${lr}_${model}_seed${i}_${k}_$gpu${gpu}_${j}_batch${b} --loss $k --opt ams --model $model --schedule decay --modality $modality --decay 0.01 2>&1>logs/${model}_seed${i}_${k}_$gpu${gpu}_${j}_batch${b}_lr_${lr}.log
                #) & 
                echo "python ../train_unet.py --seed $seed -gpu $gpu --lr $lr --shape $j -batch $b --model_dir lr_${lr}_${model}_seed${i}_${k}_$gpu${gpu}_${j}_batch${b} --loss $k --opt ams --model $model --schedule decay --modality $modality --decay 0.01 2>&1>logs/${model}_seed${i}_${k}_$gpu${gpu}_${j}_batch${b}_lr_${lr}.log"
              let gpu++
              if [[ $gpu -gt 2 ]];then
                 gpu=0
              fi
           done
           wait
       # done
       # done
   done
}

train

