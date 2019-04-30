#!/bin/bash -ex

merge_niis() {
    mkdir -p results
    mkdir -p tfrecords
    rm -f preprocess_R.commands
    while read line;do
        field=($line)
        echo Rscript preprocess.R ${field[1]} ${field[2]} ${field[3]} ${field[4]} ${field[5]} ${field[0]}  >> preprocess_R.commands
    #done < <(cut -f 2,4-8 results/data_for_model.xls | sed 1d)
    done < <(cut -f 2,4-8 results/data_for_model_manual_fixed_resample_failure.xls | sed 1d)
    parallel -j 28 < preprocess_R.commands
}

write_tfrecord() {
    #rm -f tfrecord.commands
    #for i in nii/*__not_iso_merge_chan.nii.gz; do
    #    echo $i
    #    echo python generate_tfrecord.py --infile $i >> tfrecord.commands
    #done
    #parallel -j 16 < tfrecord.commands

    #rm -f tfrecord.commands.lower
    #for i in nii/*lower*nii.gz; do
    #    echo $i
    #    echo python generate_tfrecord.py --infile $i >> tfrecord.commands.lower
    #done
    #parallel -j 16 < tfrecord.commands.lower

    rm -f tfrecord.commands.136
    for i in nii/*__136__*nii.gz; do
        echo $i
        echo python generate_tfrecord.py --infile $i >> tfrecord.commands.136
    done
    parallel -j 16 < tfrecord.commands.136
}

train_models() {
    b=32
    #mul=1
    modal=0_1_2_3_4
    l=0.0001
    loss=bce
    seed=99
#for modal in 0_1_2_3_4 0_1_2_3 0_1_2 0_1 0; do
##for modal in 0_1_2_3 0_1_2 0_1 0; do
#    time python train.py -seed $seed -loss $loss -gpu $gpu -batch $b -multilabel $mul -model $m -modal $modal -lr $l --model_dir ${m}_gpu${gpu}_batch${b}_label${mul}_modal${modal}_lr${l}_loss${loss}_seed${seed}
#done
    #gpu=2
    #m=vgg
    #for modal in 0_1_2_3_4 0_1_2_3 0_1_2 0_1 0; do
    #    time python train.py -seed $seed -loss $loss -gpu $gpu -batch $b -multilabel $mul -model $m -modal $modal -lr $l --model_dir ${m}_gpu${gpu}_batch${b}_label${mul}_modal${modal}_lr${l}_loss${loss}_seed${seed}
    #done
    #gpu=2
    #m=vgg
    gpu=0
    m=inception
    for mul in 1 2 3 4 5 6; do
        for modal in 0_1_2_3_4 0_1_2_3 0_1_2 0_1 0; do
            #time python train.py -seed $seed -loss $loss -gpu $gpu -batch $b -multilabel $mul -model $m -modal $modal -lr $l --model_dir ${m}_gpu${gpu}_batch${b}_label${mul}_modal${modal}_lr${l}_loss${loss}_seed${seed}_newstart
            time python train.py -seed $seed -loss $loss -gpu $gpu -batch $b -multilabel $mul -model $m -modal $modal -lr $l --model_dir ${m}_gpu${gpu}_batch${b}_label${mul}_modal${modal}_lr${l}_loss${loss}_seed${seed}
        done
    done
}

main() {
   #write_tfrecord
   train_models
}

main
